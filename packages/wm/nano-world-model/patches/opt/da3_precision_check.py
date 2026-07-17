#!/usr/bin/env python
"""DA3-LARGE fp32 vs bf16 A/B validation.

DA3 is a multi-view transformer (attention/matmul-bound), so bf16 autocast
should accelerate it far more than it helped the conv-bound VAE decode.
The risk is geometric: depth + camera pose estimation must not degrade.

This reuses the pipeline's own preprocessing (extract_frames,
restore_native_resolution, load_da3_model, run_inference) so the comparison
matches production exactly, then reports:
  - forward wall time fp32 vs bf16 (speedup)
  - depth agreement: median/p95 relative error, per-frame
  - camera agreement: intrinsics + extrinsics (translation/rotation) drift
  - kept-point count under the same confidence percentile
and writes a side-by-side depth PNG (fp32 | bf16 | |error|) for eyeballing.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

SCRIPTS_DIR = "/repos/nanowm/src/scripts"
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import video_to_pointcloud as v2p  # noqa: E402


def _rel_depth_err(a: np.ndarray, b: np.ndarray) -> dict:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    denom = np.maximum(np.abs(a), 1e-6)
    rel = np.abs(a - b) / denom
    return {
        "median_rel": float(np.median(rel)),
        "p95_rel": float(np.percentile(rel, 95)),
        "max_rel": float(np.max(rel)),
        "mean_abs": float(np.mean(np.abs(a - b))),
    }


def _rot_angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    # geodesic angle between two rotation matrices
    R = Ra @ Rb.T
    cos = (np.trace(R) - 1.0) / 2.0
    cos = float(np.clip(cos, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", default="depth-anything/DA3-LARGE-1.1")
    ap.add_argument("--native_h", type=int, default=150)
    ap.add_argument("--native_w", type=int, default=280)
    ap.add_argument("--max_frames", type=int, default=20)
    ap.add_argument("--frame_step", type=int, default=2)
    ap.add_argument("--process_res", type=int, default=504)
    ap.add_argument("--conf_threshold", type=float, default=40.0)
    ap.add_argument("--max_points", type=int, default=1_000_000)
    ap.add_argument("--out_dir", default="/outputs/_inspect/da3_bf16")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    frames = v2p.extract_frames(args.video, args.max_frames, args.frame_step)
    frames = v2p.restore_native_resolution(frames, args.native_h, args.native_w)

    model = v2p.load_da3_model(args.model, "cuda")

    def run(tag, autocast):
        torch.cuda.synchronize()
        t0 = time.time()
        if autocast:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = v2p.run_inference(
                    model, frames, str(out / f"exp_{tag}"),
                    args.process_res, args.conf_threshold, args.max_points,
                )
        else:
            pred = v2p.run_inference(
                model, frames, str(out / f"exp_{tag}"),
                args.process_res, args.conf_threshold, args.max_points,
            )
        torch.cuda.synchronize()
        dt = time.time() - t0
        return pred, dt

    # warm caches with fp32 first, then time both back-to-back
    pred32, t32 = run("fp32", autocast=False)
    pred16, t16 = run("bf16", autocast=True)

    depth32 = np.asarray(pred32.depth, dtype=np.float32)
    depth16 = np.asarray(pred16.depth, dtype=np.float32)
    K32 = np.asarray(pred32.intrinsics, dtype=np.float64)
    K16 = np.asarray(pred16.intrinsics, dtype=np.float64)
    E32 = np.asarray(pred32.extrinsics, dtype=np.float64)
    E16 = np.asarray(pred16.extrinsics, dtype=np.float64)

    n = depth32.shape[0]
    per_frame = []
    for i in range(n):
        d = _rel_depth_err(depth32[i], depth16[i])
        t_drift = float(np.linalg.norm(E32[i][:3, 3] - E16[i][:3, 3]))
        rot = _rot_angle_deg(E32[i][:3, :3], E16[i][:3, :3])
        per_frame.append({"frame": i, **d, "trans_drift": t_drift, "rot_deg": rot})

    # keep-count under the same conf percentile per frame
    def kept(pred):
        conf = np.asarray(pred.conf, dtype=np.float32)
        tot = 0
        for i in range(conf.shape[0]):
            thr = np.percentile(conf[i], args.conf_threshold)
            tot += int((conf[i] >= thr).sum())
        return tot

    k32, k16 = kept(pred32), kept(pred16)

    summary = {
        "video": args.video,
        "model": args.model,
        "frames": n,
        "time_fp32_s": round(t32, 2),
        "time_bf16_s": round(t16, 2),
        "speedup": round(t32 / max(t16, 1e-6), 3),
        "depth_median_rel_over_frames": round(float(np.median([p["median_rel"] for p in per_frame])), 5),
        "depth_p95_rel_max": round(float(np.max([p["p95_rel"] for p in per_frame])), 5),
        "trans_drift_max": round(float(np.max([p["trans_drift"] for p in per_frame])), 6),
        "rot_deg_max": round(float(np.max([p["rot_deg"] for p in per_frame])), 4),
        "intrinsics_max_abs": round(float(np.max(np.abs(K32 - K16))), 4),
        "kept_points_fp32": k32,
        "kept_points_bf16": k16,
        "kept_ratio": round(k16 / max(k32, 1), 4),
    }

    (out / "da3_bf16_summary.json").write_text(json.dumps({"summary": summary, "per_frame": per_frame}, indent=2))

    # side-by-side depth PNG for a few frames: fp32 | bf16 | |error|
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        idxs = sorted(set([0, n // 2, n - 1]))
        fig, axes = plt.subplots(len(idxs), 3, figsize=(12, 3.2 * len(idxs)))
        if len(idxs) == 1:
            axes = axes[None, :]
        for r, i in enumerate(idxs):
            vmin = float(min(depth32[i].min(), depth16[i].min()))
            vmax = float(max(depth32[i].max(), depth16[i].max()))
            axes[r, 0].imshow(depth32[i], cmap="turbo", vmin=vmin, vmax=vmax)
            axes[r, 0].set_title(f"frame {i}  fp32")
            axes[r, 1].imshow(depth16[i], cmap="turbo", vmin=vmin, vmax=vmax)
            axes[r, 1].set_title("bf16")
            err = np.abs(depth32[i] - depth16[i])
            im = axes[r, 2].imshow(err, cmap="magma")
            axes[r, 2].set_title(f"|err| (max {err.max():.3f})")
            plt.colorbar(im, ax=axes[r, 2], fraction=0.046, pad=0.04)
            for c in range(3):
                axes[r, c].axis("off")
        plt.tight_layout()
        fig.savefig(out / "da3_bf16_depth_compare.png", dpi=110)
        print(f"Saved {out / 'da3_bf16_depth_compare.png'}")
    except Exception as e:  # noqa: BLE001
        print(f"[warn] depth PNG failed: {e}")

    print("=" * 60)
    print("DA3 fp32 vs bf16 SUMMARY")
    print(json.dumps(summary, indent=2))
    print("=" * 60)


if __name__ == "__main__":
    main()
