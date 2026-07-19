#!/usr/bin/env python3
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Minimal, idempotent port patches for the mees/calvin simulator stack (calvin_env +
# calvin_agent) so it runs headless on ROCm/gfx1151 with the base image's modern Python 3.12 /
# numpy 2.x / pybullet 3.2.7 (rule 2.1: rely on upstream, patch only what the platform needs).
# String-replacement based so it survives small upstream drift. Run: `python calvin_port.py /repos/calvin`.
#
# What it changes and why:
#  1. calvin_env/calvin_env/camera/camera.py — process_rgbd() reshapes pybullet's getCameraImage
#     pixel buffer to (H,W,4) and slices RGB. Under pybullet 3.2.7 + numpy 2.x the buffer comes
#     back as python ints -> the array is int64, so PIL/torch downstream (GR-1's step() does
#     Image.fromarray) raise "Cannot handle this data type". Cast the RGB slice to uint8 (the
#     values are already 0-255), matching what the training data and the model preprocessing expect.
import sys, os

# (1) uint8 camera RGB (pybullet 3.2.7 + numpy 2.x returns int64 pixels) ---------------------
CAM_OLD = "        rgb_img = rgb[:, :, :3]"
CAM_NEW = "        rgb_img = rgb[:, :, :3].astype(np.uint8)  # ROCm/numpy2 port: pybullet 3.2.7 returns int64 pixels"


def replace_in_file(path, old, new, tag):
    if not os.path.isfile(path):
        print(f"  [WARN] missing: {path}")
        return False
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    if new in src and old not in src:
        print(f"  [skip] already patched ({tag}): {path}")
        return True
    if old not in src:
        print(f"  [WARN] target not found ({tag}, upstream drift?): {path}")
        return False
    with open(path, "w", encoding="utf-8") as f:
        f.write(src.replace(old, new))
    print(f"  [ok] patched {tag}: {path}")
    return True


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "/repos/calvin"
    replace_in_file(os.path.join(root, "calvin_env/calvin_env/camera/camera.py"),
                    CAM_OLD, CAM_NEW, "camera-uint8")
    print("calvin_port.py: done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
