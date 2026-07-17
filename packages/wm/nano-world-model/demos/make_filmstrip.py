#!/usr/bin/env python
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Sample N evenly-spaced frames from a video into a single small filmstrip PNG for quick
inline inspection on the laptop (keeps artifacts/ small, rule 4)."""
import sys, argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--width", type=int, default=1100)
    args = ap.parse_args()

    import imageio.v2 as imageio
    rd = imageio.get_reader(args.video)
    frames = [f for f in rd]
    if not frames:
        print("no frames in", args.video); return 1
    idx = np.linspace(0, len(frames) - 1, min(args.cols, len(frames))).astype(int)
    sel = [frames[i] for i in idx]
    h, w = sel[0].shape[:2]
    strip = np.concatenate([np.pad(f, ((0, 0), (0, 2), (0, 0)), constant_values=255) for f in sel], axis=1)

    from PIL import Image
    im = Image.fromarray(strip)
    scale = args.width / im.width
    im = im.resize((args.width, max(1, int(im.height * scale))))
    im.save(args.out)
    print("wrote", args.out, im.size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
