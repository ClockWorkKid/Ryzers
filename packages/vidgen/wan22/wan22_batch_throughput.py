#!/usr/bin/env python3

# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Batched multi-video throughput probe for WAN 2.2 TI2V-5B on Strix Halo.

Playbook lever 2.4 (increase effective batch M): generate N videos in one
batched DiT forward (the model natively concatenates a list of latents into a
batch) driven by a single lockstep UniPC scheduler over a [N, C, T, H, W]
sample, then decode all N with the batched tiled VAE. This does NOT reduce
single-video latency; it measures throughput (videos/hour) at higher GPU/
bandwidth utilization vs. running the videos sequentially.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import sys
import time
from datetime import datetime

import torch


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Batched TI2V-5B throughput probe.")
    p.add_argument("--prompt", required=True)
    p.add_argument("--num_videos", type=int, default=2)
    p.add_argument("--size", default="1280x704")
    p.add_argument("--frame_num", type=int, default=17)
    p.add_argument("--sample_steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=26080401)
    p.add_argument("--ckpt_dir", default="/models/Wan2.2-TI2V-5B")
    p.add_argument("--output_dir", default="/outputs")
    p.add_argument("--save_prefix", default="batch")
    p.add_argument("--tile_h", type=int, default=8)
    p.add_argument("--tile_w", type=int, default=12)
    p.add_argument("--stride_h", type=int, default=8)
    p.add_argument("--stride_w", type=int, default=12)
    p.add_argument("--crop_margin", type=int, default=0)
    p.add_argument("--vae_tile_batch", type=int, default=int(os.environ.get("VAE_TILE_BATCH", "8") or "8"))
    p.add_argument("--benchmark_json", default="")
    p.add_argument("--save_videos", action="store_true", help="Write decoded mp4s (else time-only).")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
    width, height = (int(v) for v in args.size.lower().replace("*", "x").split("x"))
    n = max(1, args.num_videos)

    sys.path.insert(0, "/ryzers")
    from apply_gfx1151_speedups import apply_gfx1151_speedups
    from generate_ti2v_strix import _patch_tiled_vae_decode

    import wan
    from wan.configs import WAN_CONFIGS
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
    from wan.utils.utils import masks_like, save_video

    cfg = WAN_CONFIGS["ti2v-5B"]

    load_t0 = time.perf_counter()
    pipe = wan.WanTI2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=True,
        convert_model_dtype=True,
    )
    opt_flags = apply_gfx1151_speedups(model=pipe.model)
    vae_bf16 = os.environ.get("WAN22_VAE_BF16", "0") != "0"
    if vae_bf16:
        pipe.vae.dtype = torch.bfloat16
    _patch_tiled_vae_decode(
        pipe.vae, args.tile_h, args.tile_w, args.stride_h, args.stride_w,
        args.crop_margin, tile_batch=args.vae_tile_batch,
    )
    load_s = time.perf_counter() - load_t0
    logging.info("Loaded pipe in %.1fs; opts=%s vae_bf16=%s N=%d", load_s, opt_flags, vae_bf16, n)

    device = pipe.device
    F = args.frame_num
    target_shape = (
        pipe.vae.model.z_dim,
        (F - 1) // pipe.vae_stride[0] + 1,
        height // pipe.vae_stride[1],
        width // pipe.vae_stride[2],
    )
    import math

    seq_len = math.ceil(
        (target_shape[2] * target_shape[3]) / (pipe.patch_size[1] * pipe.patch_size[2])
        * target_shape[1] / pipe.sp_size
    ) * pipe.sp_size

    # Text encode once (constant across the batch) + null prompt.
    context = pipe.text_encoder([args.prompt], torch.device("cpu"))
    context_null = pipe.text_encoder([pipe.sample_neg_prompt], torch.device("cpu"))
    context = [t.to(device) for t in context]
    context_null = [t.to(device) for t in context_null]
    conds = [context[0]] * n
    nulls = [context_null[0]] * n

    # Independent noise per video (different seed → distinct samples).
    noises = []
    for i in range(n):
        g = torch.Generator(device=device)
        g.manual_seed(args.seed + i)
        noises.append(torch.randn(*target_shape, dtype=torch.float32, device=device, generator=g))

    scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=pipe.num_train_timesteps, shift=1, use_dynamic_shifting=False
    )
    scheduler.set_timesteps(args.sample_steps, device=device, shift=cfg.sample_shift)
    timesteps = scheduler.timesteps

    latents = torch.stack(noises, dim=0)  # [N, C, T, H, W]
    _, mask2 = masks_like([noises[0]], zero=False)
    guide = cfg.sample_guide_scale

    pipe.model.to(device)
    torch.cuda.empty_cache()

    gen_t0 = time.perf_counter()
    dit_s = 0.0
    with torch.amp.autocast("cuda", dtype=pipe.param_dtype), torch.no_grad():
        for t in timesteps:
            ts = torch.stack([t])
            temp_ts = (mask2[0][0][:, ::2, ::2] * ts).flatten()
            temp_ts = torch.cat([temp_ts, temp_ts.new_ones(seq_len - temp_ts.size(0)) * ts])
            timestep = temp_ts.unsqueeze(0).repeat(n, 1)  # [N, seq_len]

            x_list = [latents[i] for i in range(n)]

            d0 = time.perf_counter()
            out_cond = pipe.model(x_list, t=timestep, context=conds, seq_len=seq_len)
            out_uncond = pipe.model(x_list, t=timestep, context=nulls, seq_len=seq_len)
            torch.cuda.synchronize()
            dit_s += time.perf_counter() - d0

            noise_cond = torch.stack(out_cond, dim=0)
            noise_uncond = torch.stack(out_uncond, dim=0)
            noise_pred = noise_uncond + guide * (noise_cond - noise_uncond)

            step_out = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            latents = step_out

    x0 = [latents[i] for i in range(n)]
    vae_t0 = time.perf_counter()
    videos = pipe.vae.decode(x0)
    torch.cuda.synchronize()
    vae_s = time.perf_counter() - vae_t0
    gen_s = time.perf_counter() - gen_t0

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved = []
    if args.save_videos:
        for i, v in enumerate(videos):
            f = out_dir / f"{args.save_prefix}_N{n}_{i}_{stamp}.mp4"
            save_video(tensor=v[None], save_file=str(f), fps=cfg.sample_fps, nrow=1,
                       normalize=True, value_range=(-1, 1))
            saved.append(str(f))

    per_video = gen_s / n
    logging.info(
        "BATCH N=%d: generate=%.1fs dit=%.1fs vae=%.1fs -> %.1fs/video (%.3f videos/min)",
        n, gen_s, dit_s, vae_s, per_video, 60.0 / per_video,
    )

    if args.benchmark_json:
        bench = {
            "mode": "batch_throughput",
            "num_videos": n,
            "size": [width, height],
            "frame_num": F,
            "sample_steps": args.sample_steps,
            "vae_bf16": vae_bf16,
            "vae_tile_batch": args.vae_tile_batch,
            "optimizations": opt_flags,
            "seconds": {
                "load": load_s,
                "generate": gen_s,
                "dit_seconds": dit_s,
                "vae_decode_seconds": vae_s,
                "per_video": per_video,
            },
            "videos_per_min": 60.0 / per_video,
            "saved": saved,
            "torch": torch.__version__,
            "hip": getattr(torch.version, "hip", None),
        }
        pathlib.Path(args.benchmark_json).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.benchmark_json).write_text(json.dumps(bench, indent=2) + "\n")
        logging.info("Wrote %s", args.benchmark_json)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
