#!/usr/bin/env python
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
Compose a temporally-aligned side-by-side rollout collage:
    [ GT | fp32 | fp32-trim | bf16 | bf16-trim ]
Each panel is the decoded generated rollout for one variant (same sample + seed), with a labeled
header bar. Frames are synced to the shortest clip. GT is kept leftmost (workspace viz convention).
"""
import os, argparse
import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

BAR_H = 22
SEP = 3  # white separator px between panels


def _font(sz=15):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def read(path):
    r = imageio.get_reader(path)
    frames = [f[..., :3] for f in r]
    r.close()
    return np.stack(frames)


def header(width, text, font):
    bar = Image.new("RGB", (width, BAR_H), (18, 18, 18))
    d = ImageDraw.Draw(bar)
    try:
        w = d.textlength(text, font=font)
    except Exception:
        w = len(text) * 7
    d.text(((width - w) // 2, 3), text, fill=(240, 240, 240), font=font)
    return np.asarray(bar)


def panel(frame, hdr):
    return np.concatenate([hdr, frame], axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--labels", nargs="+", required=True, help="label=path pairs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=6)
    args = ap.parse_args()
    font = _font()

    clips = [("GT", read(args.gt))]
    for lp in args.labels:
        lab, path = lp.split("=", 1)
        clips.append((lab, read(path)))

    n = min(len(c[1]) for c in clips)
    H = min(c[1].shape[1] for c in clips)
    W = min(c[1].shape[2] for c in clips)

    hdrs = [header(W, lab, font) for lab, _ in clips]
    sep_col = np.full((BAR_H + H, SEP, 3), 255, dtype=np.uint8)

    writer = imageio.get_writer(args.out, fps=args.fps, quality=9)
    for i in range(n):
        cols = []
        for (lab, clip), hdr in zip(clips, hdrs):
            f = clip[i, :H, :W]
            cols.append(panel(f, hdr))
            cols.append(sep_col)
        row = np.concatenate(cols[:-1], axis=1)
        writer.append_data(row.astype(np.uint8))
    writer.close()
    print(f"[collage] wrote {args.out}  ({n} frames, {len(clips)} panels, {row.shape[1]}x{row.shape[0]})")


if __name__ == "__main__":
    raise SystemExit(main())
