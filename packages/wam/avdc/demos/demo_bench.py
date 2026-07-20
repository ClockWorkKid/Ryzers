# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Phase 4 inference micro-benchmark for the AVDC video-diffusion model on gfx1151.

Times one Meta-World video generation (DDIM sampling of the 128x128x7-frame 3D-UNet) under the
env-gated optimizations in avdc_optim (AVDC_AMP / AVDC_COMPILE). Reports s/gen and s/step; the
first (warmup) generation pays MIOpen tuning / torch.compile cost and is excluded. Use to compare
baseline vs optimized on the same hardware.
"""
import argparse, json, os, time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default=os.environ.get("CKPT_DIR", "/models/metaworld"))
    ap.add_argument("--milestone", type=int, default=int(os.environ.get("MILESTONE") or "24"))
    ap.add_argument("--sample-steps", type=int, default=int(os.environ.get("SAMPLE_STEPS") or "8"))
    ap.add_argument("--iters", type=int, default=int(os.environ.get("ITERS") or "3"))
    ap.add_argument("--out", default=os.path.join(os.environ.get("OUT_DIR", "/outputs"), "bench"))
    args = ap.parse_args()

    import torch
    from flowdiffusion.inference_utils import get_video_model, pred_video
    import avdc_optim

    m = get_video_model(ckpts_dir=args.ckpt_dir, milestone=args.milestone, timestep=args.sample_steps)
    tags = avdc_optim.apply(m)

    frame = (np.random.rand(240, 320, 3) * 255).astype("uint8")
    _ = pred_video(m, frame, "door open")            # warmup (tuning/compile)
    torch.cuda.synchronize()
    times = []
    for _ in range(args.iters):
        t0 = time.time()
        _ = pred_video(m, frame, "door open")
        torch.cuda.synchronize()
        times.append(time.time() - t0)
    gen = float(np.mean(times))
    res = {"optims": tags or ["baseline"], "sample_steps": args.sample_steps,
           "s_per_gen": round(gen, 3), "s_per_step": round(gen / args.sample_steps, 4),
           "iters": args.iters}
    print("BENCH", json.dumps(res))
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "bench.json"), "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
