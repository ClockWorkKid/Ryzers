#!/usr/bin/env python
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
Offline (headless, CPU) renderer for the video->3D artifacts. Loads the DA3 point clouds for the
real GT rollout and the imagined full-stack rollout, renders each from two viewpoints, and composes
a single compact comparison PNG (GT on the left, imagined on the right -- workspace viz convention),
with a strip of aligned RGB|depth frames underneath. Keeps one small pullable artifact instead of
the multi-MB PLY / depth-PNG galleries.
"""
import os, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plyfile import PlyData
from PIL import Image


def load_ply(path, maxn=50000, seed=0, gamma=0.5):
    v = PlyData.read(path)["vertex"].data
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
    rgb = np.stack([v["red"], v["green"], v["blue"]], 1).astype(np.float32) / 255.0
    # csgo footage is dark -> gamma-brighten so the point-cloud structure reads on a white bg.
    rgb = np.clip(rgb, 0, 1) ** gamma
    if len(xyz) > maxn:
        idx = np.random.default_rng(seed).choice(len(xyz), maxn, replace=False)
        xyz, rgb = xyz[idx], rgb[idx]
    return xyz, rgb


def lims(xyz):
    lo, hi = np.percentile(xyz, 1, 0), np.percentile(xyz, 99, 0)
    return lo, hi


def scatter(ax, xyz, rgb, elev, azim, title):
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=np.clip(rgb, 0, 1), s=1.4, marker=".", linewidths=0)
    lo, hi = lims(xyz)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    try:
        ax.set_box_aspect(hi - lo)
    except Exception:
        pass
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off(); ax.set_title(title, fontsize=10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/outputs/video3d")
    ap.add_argument("--gt", default="gt_csgo")
    ap.add_argument("--gen", default="gen_csgo")
    ap.add_argument("--frames", type=int, nargs="+", default=[2, 5, 8])
    ap.add_argument("--out", default="/outputs/video3d/video3d_compare.png")
    args = ap.parse_args()

    gt_xyz, gt_rgb = load_ply(os.path.join(args.root, args.gt, "scene.ply"))
    gn_xyz, gn_rgb = load_ply(os.path.join(args.root, args.gen, "scene.ply"))
    views = [(-65, -90), (-25, -55)]

    fig = plt.figure(figsize=(11, 9), dpi=110)
    fig.suptitle("csgo video->3D (Depth Anything 3)  |  left: REAL GT rollout   right: IMAGINED full-stack rollout",
                 fontsize=11, y=0.99)

    # top 2 rows: point clouds, GT (col 0) vs imagined (col 1), two viewpoints (rows)
    for r, (el, az) in enumerate(views):
        axl = fig.add_subplot(3, 2, 2 * r + 1, projection="3d")
        scatter(axl, gt_xyz, gt_rgb, el, az, f"GT point cloud  (view {r+1})")
        axr = fig.add_subplot(3, 2, 2 * r + 2, projection="3d")
        scatter(axr, gn_xyz, gn_rgb, el, az, f"imagined point cloud  (view {r+1})")

    # bottom row: aligned RGB|depth strips (from the DA3 depth-vis triptychs), GT vs imagined
    def strip(tag):
        imgs = []
        for f in args.frames:
            p = os.path.join(args.root, tag, "scene_depth_vis", f"frame_{f:04d}.png")
            if os.path.exists(p):
                im = Image.open(p).convert("RGB")
                im = im.resize((im.width // 2, im.height // 2))
                imgs.append(np.asarray(im))
        if not imgs:
            return None
        h = min(i.shape[0] for i in imgs)
        return np.concatenate([i[:h] for i in imgs], axis=1)

    for c, tag in ((0, args.gt), (1, args.gen)):
        s = strip(tag)
        ax = fig.add_subplot(3, 2, 5 + c)
        ax.set_axis_off()
        ax.set_title(("GT" if c == 0 else "imagined") + " RGB|depth|conf  (frames " +
                     ",".join(map(str, args.frames)) + ")", fontsize=9)
        if s is not None:
            ax.imshow(s)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(args.out, bbox_inches="tight")
    print(f"[render] wrote {args.out}  ({os.path.getsize(args.out)//1024} KB)")


if __name__ == "__main__":
    raise SystemExit(main())
