#!/usr/bin/env python3

# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""WAN 2.2 TI2V-5B generation entry point for Strix Halo ROCm systems."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import pathlib
import random
import re
import sys
import time
import types
from datetime import datetime

import torch
from PIL import Image
from tqdm.auto import tqdm


UPSAMPLE = 16
SUPPORTED_SIZES = {
    "1280x704": (1280, 704),
    "1280*704": (1280, 704),
    "704x1280": (704, 1280),
    "704*1280": (704, 1280),
}


def _starts_for(length: int, tile: int, stride: int) -> list[int]:
    if length <= tile:
        return [0]
    starts = list(range(0, max(length - tile + 1, 1), stride))
    final = length - tile
    if starts[-1] != final:
        starts.append(final)
    return starts


def _crop_bounds(start: int, end: int, full: int, margin: int) -> tuple[int, int]:
    crop_start = margin if start > 0 else 0
    crop_end = (end - start) - (margin if end < full else 0)
    if crop_end <= crop_start:
        return 0, end - start
    return crop_start, crop_end


def _feather_weight(
    hpx: int,
    wpx: int,
    ramp_h: int,
    ramp_w: int,
    top: bool,
    bottom: bool,
    left: bool,
    right: bool,
    device,
) -> "torch.Tensor":
    """Linear feather (Hann-like) window for seamless tile blending.

    Weights ramp 0->1 across the overlap region on *interior* edges only, so
    adjacent tiles cross-fade instead of hard-stitching. Outer image edges stay
    at 1.0 (no darkening). With overlap == ramp, opposing linear ramps sum to ~1,
    which both hides tile seams AND down-weights the artifact-prone tile borders
    where the VAE decoder lacked cross-tile latent context.
    """
    wy = torch.ones(hpx, device=device, dtype=torch.float32)
    wx = torch.ones(wpx, device=device, dtype=torch.float32)
    if ramp_h > 0:
        up = torch.linspace(1.0 / (ramp_h + 1), 1.0, ramp_h, device=device)
        if top:
            wy[:ramp_h] = up
        if bottom:
            wy[-ramp_h:] = up.flip(0)
    if ramp_w > 0:
        up = torch.linspace(1.0 / (ramp_w + 1), 1.0, ramp_w, device=device)
        if left:
            wx[:ramp_w] = up
        if right:
            wx[-ramp_w:] = up.flip(0)
    return wy[:, None] * wx[None, :]


def _frame_count_for_seconds(seconds: float, fps: int) -> int:
    target = max(1, int(math.ceil(seconds * fps)))
    if target <= 1:
        return 1
    return 4 * int(math.ceil((target - 1) / 4)) + 1


def _parse_size(value: str) -> tuple[int, int]:
    normalized = value.lower().replace(" ", "")
    if normalized not in SUPPORTED_SIZES:
        choices = ", ".join(sorted(set(SUPPORTED_SIZES)))
        raise argparse.ArgumentTypeError(f"unsupported TI2V size {value!r}; choose one of: {choices}")
    return SUPPORTED_SIZES[normalized]


def _slugify_prompt(prompt: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", prompt.strip().lower()).strip("_")
    return slug[:48] or "prompt"


def _decode_tile_batch(vae, tiles: list) -> list:
    """Decode a list of same-shape latent tiles in one batched VAE call.

    The Wan2.2 VAE decoder is batch-safe (batch dim is preserved through the
    causal-conv temporal loop and cleared per call), so stacking tiles into one
    [B, C, T, H, W] decode maximizes GPU/bandwidth utilization vs. one-at-a-time.
    """
    batch = torch.stack([t.contiguous() for t in tiles], dim=0)
    with torch.amp.autocast("cuda", dtype=vae.dtype):
        out = vae.model.decode(batch, vae.scale).float().clamp_(-1, 1)
    return [out[i] for i in range(out.shape[0])]


def _patch_tiled_vae_decode(
    vae,
    tile_h: int,
    tile_w: int,
    stride_h: int,
    stride_w: int,
    crop_margin: int,
    tile_batch: int = 1,
) -> None:
    original_decode = vae.decode

    def tiled_decode(self_vae, zs):
        if not isinstance(zs, list):
            raise TypeError("zs should be a list")

        decoded_all = []
        for z in zs:
            _, _latent_t, height, width = z.shape
            h_starts = _starts_for(height, tile_h, stride_h)
            w_starts = _starts_for(width, tile_w, stride_w)
            logging.info(
                "Tiled VAE decode: latent=%s tile=%sx%s stride=%sx%s crop=%s grid=%sx%s batch=%s dtype=%s",
                tuple(z.shape),
                tile_h,
                tile_w,
                stride_h,
                stride_w,
                crop_margin,
                len(h_starts),
                len(w_starts),
                tile_batch,
                vae.dtype,
            )

            coords = [
                (h0, min(h0 + tile_h, height), w0, min(w0 + tile_w, width))
                for h0 in h_starts
                for w0 in w_starts
            ]
            total_tiles = len(coords)
            values = None
            weights = None
            effective_batch = max(1, tile_batch)

            with tqdm(total=total_tiles, desc="VAE tiles", unit="tile") as pbar:
                for start in range(0, total_tiles, effective_batch):
                    chunk = coords[start:start + effective_batch]
                    if effective_batch <= 1:
                        h0, h1, w0, w1 = chunk[0]
                        tile_outs = [original_decode([z[:, :, h0:h1, w0:w1].contiguous()])[0]]
                    else:
                        tiles = [z[:, :, h0:h1, w0:w1] for (h0, h1, w0, w1) in chunk]
                        tile_outs = _decode_tile_batch(vae, tiles)

                    ramp_h = max(0, (tile_h - stride_h)) * UPSAMPLE
                    ramp_w = max(0, (tile_w - stride_w)) * UPSAMPLE
                    for (h0, h1, w0, w1), tile_out in zip(chunk, tile_outs):
                        if values is None:
                            values = torch.zeros(
                                (tile_out.shape[0], tile_out.shape[1], height * UPSAMPLE, width * UPSAMPLE),
                                device=tile_out.device,
                                dtype=torch.float32,
                            )
                            weights = torch.zeros_like(values)

                        hpx = (h1 - h0) * UPSAMPLE
                        wpx = (w1 - w0) * UPSAMPLE
                        wmap = _feather_weight(
                            hpx,
                            wpx,
                            ramp_h,
                            ramp_w,
                            top=h0 > 0,
                            bottom=h1 < height,
                            left=w0 > 0,
                            right=w1 < width,
                            device=tile_out.device,
                        )
                        oh0, oh1 = h0 * UPSAMPLE, h1 * UPSAMPLE
                        ow0, ow1 = w0 * UPSAMPLE, w1 * UPSAMPLE
                        tile_f = tile_out.float()
                        values[:, :, oh0:oh1, ow0:ow1] += tile_f * wmap
                        weights[:, :, oh0:oh1, ow0:ow1] += wmap
                        pbar.update(1)

            if torch.any(weights == 0):
                raise RuntimeError("tiled VAE decode left uncovered pixels; reduce stride or increase tile size")
            decoded_all.append((values / weights).clamp_(-1, 1))
        return decoded_all

    vae.decode = types.MethodType(tiled_decode, vae)


def _patch_vae_encode_offload(vae) -> None:
    """Free the VAE encode residue before the DiT denoise loop (i2v OOM fix).

    Only i2v calls vae.encode(), and it runs at full 704x1280 resolution right
    before the long DiT loop. Its encoder activations linger in the HIP caching
    allocator and fragment VRAM, which starved the 121f DiT self-attention
    (rope_apply) even though plain t2v at 121f fits. We park the VAE weights on
    CPU and empty_cache after encoding, then the tiled decode brings the VAE back
    to the GPU for the final decode.
    """
    original_encode = vae.encode

    def encode_offload(self_vae, videos):
        out = original_encode(videos)
        try:
            self_vae.model.to("cpu")
        except Exception:
            pass
        torch.cuda.empty_cache()
        return out

    original_decode = vae.decode

    def decode_reload(self_vae, zs):
        self_vae.model.to(self_vae.device)
        return original_decode(zs)

    vae.encode = types.MethodType(encode_offload, vae)
    vae.decode = types.MethodType(decode_reload, vae)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate WAN 2.2 TI2V-5B video on AMD Strix Halo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--image", default="", help="Optional conditioning image for TI2V image-to-video mode.")
    parser.add_argument("--mode", choices=("auto", "t2v", "i2v"), default="auto")
    parser.add_argument("--size", type=_parse_size, default=(1280, 704), help="Upstream TI2V size: 1280x704 or 704x1280.")
    parser.add_argument("--seconds", type=float, default=None, help="Duration rounded up to a valid 4n+1 frame count.")
    parser.add_argument("--frame_num", type=int, default=None, help="Explicit frame count. Must be 4n+1.")
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--ckpt_dir", default="/models/Wan2.2-TI2V-5B")
    parser.add_argument("--output_dir", default="/outputs")
    parser.add_argument("--save_file", default="")
    parser.add_argument("--no_tiled_vae_decode", action="store_true", help="Disable the Strix Halo tiled VAE fallback.")
    # Fast-route defaults: 12x12 tiles with stride 8 (4-cell / 64px overlap) that
    # feather-blend so tile seams are invisible; feathering also down-weights the
    # artifact-prone tile borders (no crop needed). Overridable via env/CLI.
    parser.add_argument("--tile_h", type=int, default=int(os.environ.get("TILE_H", "12") or "12"))
    parser.add_argument("--tile_w", type=int, default=int(os.environ.get("TILE_W", "12") or "12"))
    parser.add_argument("--stride_h", type=int, default=int(os.environ.get("STRIDE_H", "8") or "8"))
    parser.add_argument("--stride_w", type=int, default=int(os.environ.get("STRIDE_W", "8") or "8"))
    parser.add_argument("--crop_margin", type=int, default=int(os.environ.get("CROP_MARGIN", "0") or "0"))
    parser.add_argument(
        "--vae_tile_batch",
        type=int,
        default=int(os.environ.get("VAE_TILE_BATCH", "8") or "8"),
        help="Number of same-shape VAE tiles to decode per batched call (1 = one-at-a-time).",
    )
    parser.add_argument("--benchmark_json", default="", help="Write wall-clock timing JSON to this path.")
    return parser


def _load_condition_image(path: str) -> Image.Image:
    image_path = pathlib.Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(f"conditioning image not found: {path}")
    with Image.open(image_path) as img:
        return img.convert("RGB")


def _resolve_mode(args) -> str:
    if args.mode != "auto":
        return args.mode
    return "i2v" if args.image else "t2v"


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
    if args.mode == "i2v" and not args.image:
        raise SystemExit("--mode i2v requires --image")

    sys.path.insert(0, "/ryzers")
    from apply_gfx1151_speedups import apply_gfx1151_speedups
    from wan22_playbook_opts import apply_playbook_opts, get_timings

    import wan
    from wan.configs import WAN_CONFIGS
    from wan.utils.utils import save_video

    cfg = WAN_CONFIGS["ti2v-5B"]
    width, height = args.size
    mode = _resolve_mode(args)
    cond_image = _load_condition_image(args.image) if mode == "i2v" else None
    if args.frame_num is not None and args.seconds is not None:
        raise SystemExit("choose either --frame_num or --seconds, not both")
    if args.seconds is not None:
        frame_num = _frame_count_for_seconds(args.seconds, cfg.sample_fps)
    elif args.frame_num is not None:
        frame_num = args.frame_num
    else:
        frame_num = cfg.frame_num
    if frame_num % 4 != 1:
        raise SystemExit(f"frame_num must be 4n+1 for WAN VAE temporal stride; got {frame_num}")

    seed = args.seed
    if seed < 0:
        seed = random.randint(0, sys.maxsize)
    torch.cuda.set_device(0)

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_file:
        save_file = pathlib.Path(args.save_file)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prompt_slug = _slugify_prompt(args.prompt)
        mode_tag = mode
        save_file = output_dir / f"{mode_tag}5b_{width}x{height}_f{frame_num}_s{args.sample_steps}_{seed}_{prompt_slug}_{stamp}.mp4"
    save_file.parent.mkdir(parents=True, exist_ok=True)

    logging.info(
        "Generating TI2V-5B (%s): size=%sx%s frames=%s fps=%s steps=%s seed=%s image=%s output=%s",
        mode,
        width,
        height,
        frame_num,
        cfg.sample_fps,
        args.sample_steps,
        seed,
        args.image or "(none)",
        save_file,
    )
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
    tome_r = float(os.environ.get("WAN22_TOME_R", "0") or "0")
    compile_on = os.environ.get("WAN22_TORCH_COMPILE", "0") != "0"
    if compile_on and os.environ.get("WAN22_TEXT_KV_CACHE", "0") != "0":
        logging.warning("torch.compile + KV-cache monkeypatch conflict; disabling KV cache for this run")
        os.environ["WAN22_TEXT_KV_CACHE"] = "0"

    opt_flags = apply_gfx1151_speedups(model=pipe.model)
    vae_bf16 = os.environ.get("WAN22_VAE_BF16", "1") != "0"
    if vae_bf16:
        pipe.vae.dtype = torch.bfloat16
    opt_flags["vae_bf16"] = vae_bf16
    opt_flags["vae_tile_batch"] = args.vae_tile_batch

    if tome_r > 0:
        from wan22_playbook_opts import apply_dit_tome
        opt_flags.update(apply_dit_tome(pipe.model, tome_r))
    if compile_on:
        from wan22_playbook_opts import apply_torch_compile
        opt_flags.update(apply_torch_compile(pipe))
    logging.info("Active optimizations: %s", opt_flags)
    if not args.no_tiled_vae_decode:
        _patch_tiled_vae_decode(
            pipe.vae,
            args.tile_h,
            args.tile_w,
            args.stride_h,
            args.stride_w,
            args.crop_margin,
            tile_batch=args.vae_tile_batch,
        )
    vae_offload = os.environ.get("WAN22_VAE_OFFLOAD", "1") != "0"
    if vae_offload and mode == "i2v":
        _patch_vae_encode_offload(pipe.vae)
    opt_flags["vae_offload"] = vae_offload and mode == "i2v"
    load_s = time.perf_counter() - load_t0

    gen_t0 = time.perf_counter()
    video = pipe.generate(
        args.prompt,
        img=cond_image,
        size=(width, height),
        max_area=width * height,
        frame_num=frame_num,
        shift=cfg.sample_shift,
        sample_solver="unipc",
        sampling_steps=args.sample_steps,
        guide_scale=cfg.sample_guide_scale,
        seed=seed,
        offload_model=True,
    )
    torch.cuda.synchronize()
    gen_s = time.perf_counter() - gen_t0

    save_t0 = time.perf_counter()
    save_video(
        tensor=video[None],
        save_file=str(save_file),
        fps=cfg.sample_fps,
        nrow=1,
        normalize=True,
        value_range=(-1, 1),
    )
    save_s = time.perf_counter() - save_t0
    total_s = load_s + gen_s + save_s
    logging.info(
        "Timing: load=%.2fs generate=%.2fs save=%.2fs total=%.2fs (%.3fs/frame, %.3fs/step)",
        load_s,
        gen_s,
        save_s,
        total_s,
        gen_s / max(frame_num, 1),
        gen_s / max(args.sample_steps, 1),
    )

    if args.benchmark_json:
        bench_path = pathlib.Path(args.benchmark_json)
        bench_path.parent.mkdir(parents=True, exist_ok=True)
        bench = {
            "mode": mode,
            "prompt": args.prompt,
            "image": args.image or None,
            "size": [width, height],
            "frame_num": frame_num,
            "sample_steps": args.sample_steps,
            "seed": seed,
            "save_file": str(save_file),
            "optimizations": opt_flags,
            "tile": {
                "tile_h": args.tile_h,
                "tile_w": args.tile_w,
                "stride_h": args.stride_h,
                "stride_w": args.stride_w,
                "crop_margin": args.crop_margin,
                "tiled": not args.no_tiled_vae_decode,
            },
            "seconds": {
                "load": load_s,
                "generate": gen_s,
                "save": save_s,
                "total": total_s,
                "per_frame": gen_s / max(frame_num, 1),
                "per_step": gen_s / max(args.sample_steps, 1),
                **get_timings(),
            },
            "torch": torch.__version__,
            "hip": getattr(torch.version, "hip", None),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
        bench_path.write_text(json.dumps(bench, indent=2) + "\n", encoding="utf-8")
        logging.info("Wrote benchmark JSON to %s", bench_path)

    del video, pipe
    torch.cuda.synchronize()
    logging.info("Generation complete: %s", save_file)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
