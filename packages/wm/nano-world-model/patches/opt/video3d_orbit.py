#!/usr/bin/env python
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
Headless (CPU) orbit renderer for the multi-seed video->3D worlds. For each seed's DA3 point cloud
it renders a 360-deg azimuth fly-around mp4 (a 3D world to "visit" without a live viser server), and
composes a single multi-seed montage PNG (all seeds from one canonical view). Colors are gamma-
brightened because csgo footage is dark. Point clouds / npz stay remote; only the small mp4s + the
montage are meant to be pulled.
"""
import os, glob, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from plyfile import PlyData


def load_ply(path, maxn=45000, seed=0, gamma=0.5):
    v = PlyData.read(path)["vertex"].data
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
    rgb = (np.stack([v["red"], v["green"], v["blue"]], 1).astype(np.float32) / 255.0)
    rgb = np.clip(rgb, 0, 1) ** gamma
    if len(xyz) > maxn:
        idx = np.random.default_rng(seed).choice(len(xyz), maxn, replace=False)
        xyz, rgb = xyz[idx], rgb[idx]
    lo, hi = np.percentile(xyz, 1, 0), np.percentile(xyz, 99, 0)
    return xyz, rgb, lo, hi


def _style(ax, lo, hi, elev, azim):
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    try:
        ax.set_box_aspect(hi - lo)
    except Exception:
        pass
    ax.view_init(elev=elev, azim=azim); ax.set_axis_off()


def orbit(xyz, rgb, lo, hi, out, n=48, elev=-30, dpi=100, size=5.0):
    fig = plt.figure(figsize=(size, size), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    writer = imageio.get_writer(out, fps=12, quality=8, macro_block_size=8)
    for az in np.linspace(-180, 180, n, endpoint=False):
        ax.clear()
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=rgb, s=1.3, marker=".", linewidths=0)
        _style(ax, lo, hi, elev, az)
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        writer.append_data(buf.reshape(h, w, 4)[..., :3].copy())
    writer.close(); plt.close(fig)
    print(f"[orbit] wrote {out} ({os.path.getsize(out)//1024} KB, {n} frames)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/outputs/csgo_multi")
    ap.add_argument("--out_dir", default="/outputs/csgo_multi/renders")
    ap.add_argument("--frames", type=int, default=48)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    plys = sorted(glob.glob(os.path.join(args.root, "seed*", "scene.ply")))
    if not plys:
        print("no point clouds found under", args.root); return 1
    clouds = []
    for p in plys:
        tag = p.split(os.sep)[-2]  # seedN
        xyz, rgb, lo, hi = load_ply(p)
        clouds.append((tag, xyz, rgb, lo, hi))
        orbit(xyz, rgb, lo, hi, os.path.join(args.out_dir, f"orbit_{tag}.mp4"), n=args.frames)

    # montage: all seeds from one canonical view
    ncol = min(len(clouds), 4); nrow = (len(clouds) + ncol - 1) // ncol
    fig = plt.figure(figsize=(4 * ncol, 4 * nrow), dpi=110)
    fig.suptitle("csgo imagined 3D worlds (full-stack world model -> DA3) — several seeds", fontsize=12, y=0.99)
    for i, (tag, xyz, rgb, lo, hi) in enumerate(clouds):
        ax = fig.add_subplot(nrow, ncol, i + 1, projection="3d")
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=rgb, s=1.0, marker=".", linewidths=0)
        _style(ax, lo, hi, -30, -90); ax.set_title(tag, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    mp = os.path.join(args.out_dir, "seeds_montage.png")
    fig.savefig(mp, bbox_inches="tight"); plt.close(fig)
    print(f"[montage] wrote {mp} ({os.path.getsize(mp)//1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
