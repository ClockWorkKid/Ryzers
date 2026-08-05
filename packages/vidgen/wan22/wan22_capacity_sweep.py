#!/usr/bin/env python3

# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Hardware-capacity sweep for WAN 2.2 TI2V-5B on Strix Halo (gfx1151).

Goal: decide which pipeline stages are worth *batching* (share weight/bandwidth,
sub-linear cost) vs. which should be *serialized* (already bandwidth-saturated,
linear cost). We measure:

  1. Roofline calibration on THIS device: achievable peak bf16 GEMM TFLOP/s and
     peak memory bandwidth GB/s (-> empirical ridge point).
  2. DiT forward: isolated timing across effective batch M in {1..8} at the
     production sequence length. Per-sample time vs M reveals spare capacity.
  3. VAE decode: isolated timing across tile-batch in {1..42}. Per-tile time vs
     batch reveals whether the (bandwidth-bound) decoder has any headroom.

Only JSON is written; plotting/analysis happens off-device on the laptop.
Phase boundaries are printed with epoch timestamps so an external rocm-smi
busy% sampler can be correlated per phase.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import pathlib
import sys
import time

import torch

PEAK_TFLOPS_SPEC = 37.8   # gfx1151 bf16 matrix peak (spec)
PEAK_GBS_SPEC = 230.0     # LPDDR5x memory bandwidth (spec)


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _time(fn, iters: int, warmup: int = 2) -> float:
    for _ in range(warmup):
        fn()
    _sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync()
    return (time.perf_counter() - t0) / iters


def _phase(name: str) -> None:
    logging.info("PHASE %s t=%.3f", name, time.time())


def roofline_calibration(device) -> dict:
    _phase("roofline_gemm")
    gemm = []
    for n in (1024, 2048, 4096, 8192):
        a = torch.randn(n, n, device=device, dtype=torch.bfloat16)
        b = torch.randn(n, n, device=device, dtype=torch.bfloat16)
        iters = max(3, int(2e11 / (2 * n**3)))
        t = _time(lambda: torch.mm(a, b), iters)
        tflops = 2 * n**3 / t / 1e12
        gemm.append({"n": n, "ms": t * 1e3, "tflops": tflops, "mfu": tflops / PEAK_TFLOPS_SPEC})
        del a, b
        torch.cuda.empty_cache()

    _phase("roofline_bw")
    bw = []
    for mb in (64, 256, 1024):
        nelem = mb * 1024 * 1024 // 2  # bf16
        x = torch.randn(nelem, device=device, dtype=torch.bfloat16)
        iters = max(5, int(4e10 / (nelem * 2)))
        # out = x + 1: read x, write out -> 2 * bytes traffic
        t = _time(lambda: torch.add(x, 1.0), iters)
        gbs = 2 * nelem * 2 / t / 1e9
        bw.append({"mb": mb, "ms": t * 1e3, "gbs": gbs, "mbu": gbs / PEAK_GBS_SPEC})
        del x
        torch.cuda.empty_cache()

    peak_tflops = max(g["tflops"] for g in gemm)
    peak_gbs = max(b["gbs"] for b in bw)
    return {
        "gemm": gemm,
        "bandwidth": bw,
        "achieved_peak_tflops": peak_tflops,
        "achieved_peak_gbs": peak_gbs,
        "ridge_flops_per_byte": peak_tflops * 1e12 / (peak_gbs * 1e9),
        "spec_ridge_flops_per_byte": PEAK_TFLOPS_SPEC * 1e12 / (PEAK_GBS_SPEC * 1e9),
    }


def dit_flops(model, seq_len: int, l_txt: int) -> float:
    d = model.dim
    f = model.ffn_dim
    ly = len(model.blocks)
    L = seq_len
    per_layer = (
        8 * L * d * d           # self qkvo
        + 4 * L * L * d         # self attn scores+av
        + 2 * (L + 2 * l_txt) * d * d   # cross qkv proj
        + 4 * L * l_txt * d     # cross attn
        + 4 * L * d * f         # ffn
    )
    return per_layer * ly


def sweep_dit(pipe, device, frame_num: int, batches, iters: int) -> dict:
    F = frame_num
    target_shape = (
        pipe.vae.model.z_dim,
        (F - 1) // pipe.vae_stride[0] + 1,
        704 // pipe.vae_stride[1],
        1280 // pipe.vae_stride[2],
    )
    seq_len = math.ceil(
        (target_shape[2] * target_shape[3]) / (pipe.patch_size[1] * pipe.patch_size[2])
        * target_shape[1] / pipe.sp_size
    ) * pipe.sp_size

    context = pipe.text_encoder(["a cinematic scene, high detail"], torch.device("cpu"))
    ctx0 = context[0].to(device)
    l_txt = ctx0.shape[0]
    fl = dit_flops(pipe.model, seq_len, l_txt)

    from wan.utils.utils import masks_like

    pipe.model.to(device)
    torch.cuda.empty_cache()
    results = []
    _phase(f"dit_f{F}_seq{seq_len}")
    with torch.amp.autocast("cuda", dtype=pipe.param_dtype), torch.no_grad():
        for m in batches:
            noise = torch.randn(*target_shape, dtype=torch.float32, device=device)
            _, mask2 = masks_like([noise], zero=False)
            ts = torch.tensor([1000.0], device=device)
            temp_ts = (mask2[0][0][:, ::2, ::2] * ts).flatten()
            temp_ts = torch.cat([temp_ts, temp_ts.new_ones(seq_len - temp_ts.size(0)) * ts])
            timestep = temp_ts.unsqueeze(0).repeat(m, 1)
            x_list = [noise for _ in range(m)]
            ctx_list = [ctx0 for _ in range(m)]

            try:
                t = _time(lambda: pipe.model(x_list, t=timestep, context=ctx_list, seq_len=seq_len), iters)
            except RuntimeError as e:
                results.append({"batch": m, "oom": True, "err": str(e)[:200]})
                torch.cuda.empty_cache()
                break
            peak_gb = torch.cuda.max_memory_allocated() / 1e9
            torch.cuda.reset_peak_memory_stats()
            achieved_tflops = fl * m / t / 1e12
            results.append({
                "batch": m,
                "ms": t * 1e3,
                "ms_per_sample": t * 1e3 / m,
                "achieved_tflops": achieved_tflops,
                "mfu": achieved_tflops / PEAK_TFLOPS_SPEC,
                "peak_mem_gb": peak_gb,
            })
            logging.info("DiT M=%d: %.1f ms (%.1f ms/sample) %.2f TFLOP/s mfu=%.1f%% peak=%.1fGB",
                         m, t * 1e3, t * 1e3 / m, achieved_tflops, achieved_tflops / PEAK_TFLOPS_SPEC * 100, peak_gb)
    return {"frame_num": F, "seq_len": seq_len, "l_txt": l_txt, "flops_per_forward_M1": fl, "runs": results}


def sweep_vae(pipe, device, frame_num: int, tile_batches, iters: int) -> dict:
    from generate_ti2v_strix import _decode_tile_batch

    zc = pipe.vae.model.z_dim
    Tlat = (frame_num - 1) // pipe.vae_stride[0] + 1
    Hlat, Wlat = 704 // pipe.vae_stride[1], 1280 // pipe.vae_stride[2]
    th, tw = 8, 12
    h_starts = list(range(0, max(1, Hlat - th + 1), th))
    if h_starts[-1] != Hlat - th:
        h_starts.append(Hlat - th)
    w_starts = list(range(0, max(1, Wlat - tw + 1), tw))
    if w_starts[-1] != Wlat - tw:
        w_starts.append(Wlat - tw)
    n_tiles = len(h_starts) * len(w_starts)

    z = torch.randn(zc, Tlat, Hlat, Wlat, device=device, dtype=torch.float32)
    tiles = [z[:, :, h:h + th, w:w + tw].contiguous() for h in h_starts for w in w_starts]

    results = []
    _phase(f"vae_f{frame_num}_tiles{n_tiles}")
    with torch.no_grad():
        for tb in tile_batches:
            tb = min(tb, n_tiles)

            def run_all():
                for s in range(0, n_tiles, tb):
                    _decode_tile_batch(pipe.vae, tiles[s:s + tb])

            try:
                t = _time(run_all, iters, warmup=1)
            except RuntimeError as e:
                results.append({"tile_batch": tb, "oom": True, "err": str(e)[:200]})
                torch.cuda.empty_cache()
                break
            peak_gb = torch.cuda.max_memory_allocated() / 1e9
            torch.cuda.reset_peak_memory_stats()
            results.append({
                "tile_batch": tb,
                "total_ms": t * 1e3,
                "ms_per_tile": t * 1e3 / n_tiles,
                "peak_mem_gb": peak_gb,
            })
            logging.info("VAE tile_batch=%d: %.1f ms total (%.1f ms/tile) peak=%.1fGB",
                         tb, t * 1e3, t * 1e3 / n_tiles, peak_gb)
    return {"frame_num": frame_num, "n_tiles": n_tiles, "tile_hw": [th, tw], "dtype": str(pipe.vae.dtype), "runs": results}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", default="/models/Wan2.2-TI2V-5B")
    p.add_argument("--frame_nums", default="17,33")
    p.add_argument("--dit_batches", default="1,2,3,4,6,8")
    p.add_argument("--vae_tile_batches", default="1,2,4,8,16,32,42")
    p.add_argument("--iters", type=int, default=3)
    p.add_argument("--benchmark_json", default="/outputs/capacity_sweep.json")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

    sys.path.insert(0, "/ryzers")
    from apply_gfx1151_speedups import apply_gfx1151_speedups
    import wan
    from wan.configs import WAN_CONFIGS

    frame_nums = [int(x) for x in args.frame_nums.split(",")]
    dit_batches = [int(x) for x in args.dit_batches.split(",")]
    vae_tile_batches = [int(x) for x in args.vae_tile_batches.split(",")]

    device = torch.device("cuda")
    cap = {"platform": {}, "roofline": {}, "dit": [], "vae": []}
    cap["platform"] = {
        "torch": torch.__version__,
        "hip": getattr(torch.version, "hip", None),
        "device": torch.cuda.get_device_name(0),
        "spec_peak_tflops": PEAK_TFLOPS_SPEC,
        "spec_peak_gbs": PEAK_GBS_SPEC,
    }

    cap["roofline"] = roofline_calibration(device)
    logging.info("Roofline: peak %.1f TFLOP/s, %.1f GB/s, ridge=%.1f FLOP/byte",
                 cap["roofline"]["achieved_peak_tflops"], cap["roofline"]["achieved_peak_gbs"],
                 cap["roofline"]["ridge_flops_per_byte"])

    pipe = wan.WanTI2V(config=WAN_CONFIGS["ti2v-5B"], checkpoint_dir=args.ckpt_dir,
                       device_id=0, rank=0, t5_fsdp=False, dit_fsdp=False, use_sp=False,
                       t5_cpu=True, convert_model_dtype=True)
    apply_gfx1151_speedups(model=pipe.model)
    pipe.vae.dtype = torch.bfloat16  # production config

    for fn in frame_nums:
        try:
            cap["dit"].append(sweep_dit(pipe, device, fn, dit_batches, args.iters))
        except RuntimeError as e:
            cap["dit"].append({"frame_num": fn, "error": str(e)[:300]})
            torch.cuda.empty_cache()
    for fn in frame_nums:
        cap["vae"].append(sweep_vae(pipe, device, fn, vae_tile_batches, args.iters))

    pathlib.Path(args.benchmark_json).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.benchmark_json).write_text(json.dumps(cap, indent=2) + "\n")
    logging.info("Wrote %s", args.benchmark_json)
    _phase("done")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
