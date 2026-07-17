#!/usr/bin/env python
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
Quality metrics for the optimization validation. For each run dir, reads the saved *_gen.mp4 /
*_gt.mp4 pairs and computes per-frame PSNR + SSIM (and LPIPS if available), averaged over the
GENERATED frames (skips the context prefix). Also compares the bf16 run's generated frames against
the fp32 run's generated frames (identical seed/window -> isolates the bf16 numerical effect).
"""
import os, sys, json, glob, argparse
import numpy as np
import imageio.v2 as imageio

try:
    from skimage.metrics import structural_similarity as sk_ssim
    _HAS_SK = True
except Exception:
    _HAS_SK = False

_LPIPS = None
def get_lpips():
    global _LPIPS
    if _LPIPS == "off":
        return None
    if _LPIPS is None:
        try:
            import torch, lpips
            _LPIPS = (lpips.LPIPS(net="alex").eval(), torch)
        except Exception as e:
            print(f"[metrics] LPIPS unavailable ({type(e).__name__}); skipping"); _LPIPS = "off"
            return None
    return _LPIPS


def read_video(path):
    r = imageio.get_reader(path)
    frames = [f[..., :3] for f in r]  # drop alpha if any
    r.close()
    return np.stack(frames).astype(np.float32) / 255.0  # [F,H,W,3] in [0,1]


def psnr(a, b):
    mse = np.mean((a - b) ** 2)
    return 100.0 if mse < 1e-12 else 10.0 * np.log10(1.0 / mse)


def ssim(a, b):
    if _HAS_SK:
        return float(sk_ssim(a, b, channel_axis=2, data_range=1.0))
    # simple global fallback
    mu_a, mu_b = a.mean(), b.mean(); va, vb = a.var(), b.var(); cov = ((a - mu_a) * (b - mu_b)).mean()
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return float(((2*mu_a*mu_b + c1) * (2*cov + c2)) / ((mu_a**2 + mu_b**2 + c1) * (va + vb + c2)))


def lpips_dist(a, b):
    lp = get_lpips()
    if lp is None:
        return None
    net, torch = lp
    def to_t(x):
        return torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0) * 2 - 1
    with torch.no_grad():
        return float(net(to_t(a), to_t(b)).item())


def pair_metrics(genA, genB, n_ctx):
    """Metrics between two frame stacks over generated frames only."""
    F = min(len(genA), len(genB))
    ps, ss, lp = [], [], []
    for i in range(n_ctx, F):
        ps.append(psnr(genA[i], genB[i]))
        ss.append(ssim(genA[i], genB[i]))
        d = lpips_dist(genA[i], genB[i])
        if d is not None:
            lp.append(d)
    out = dict(n_frames=F - n_ctx, psnr=float(np.mean(ps)) if ps else None,
               ssim=float(np.mean(ss)) if ss else None)
    if lp:
        out["lpips"] = float(np.mean(lp))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp32_dir", required=True)
    ap.add_argument("--bf16_dir", required=True)
    ap.add_argument("--n_context", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--domain", default="")
    args = ap.parse_args()

    result = {"domain": args.domain, "n_context": args.n_context, "samples": []}
    fp32_gens = sorted(glob.glob(os.path.join(args.fp32_dir, "*_gen.mp4")))
    for fp32_gen in fp32_gens:
        sid = os.path.basename(fp32_gen).replace("_gen.mp4", "")
        fp32_gt = os.path.join(args.fp32_dir, f"{sid}_gt.mp4")
        bf16_gen = os.path.join(args.bf16_dir, f"{sid}_gen.mp4")
        bf16_gt = os.path.join(args.bf16_dir, f"{sid}_gt.mp4")
        if not (os.path.exists(bf16_gen) and os.path.exists(fp32_gt)):
            continue
        gA, gtA = read_video(fp32_gen), read_video(fp32_gt)
        gB, gtB = read_video(bf16_gen), read_video(bf16_gt)
        rec = dict(
            sample=sid,
            fp32trim_gen_vs_gt=pair_metrics(gA, gtA, args.n_context),
            bf16trim_gen_vs_gt=pair_metrics(gB, gtB, args.n_context),
            bf16_vs_fp32_gen=pair_metrics(gB, gA, args.n_context),  # precision isolation
        )
        result["samples"].append(rec)

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
