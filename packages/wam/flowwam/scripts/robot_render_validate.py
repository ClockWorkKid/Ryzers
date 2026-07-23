# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Validate the SAPIEN RobotOnlyRenderer independently (rule 2) on AMD Strix Halo (gfx1151).
The WorldArena action-driven path renders robot-only frames from `joint_action` via SAPIEN Vulkan
offscreen; those frames drive the optical-flow conditioning. This confirms Vulkan headless render
works on ROCm before the full open-loop run. Saves a 6-frame strip to OUT_DIR for inspection."""
import glob
import os
import sys
import time

import numpy as np
from PIL import Image

FLOWWAM_REPO = os.environ.get("FLOWWAM_REPO", "/repos/flowwam")
EMB = os.environ.get("EMBODIMENT_DIR", "/models/flowwam/embodiments")
DATA = os.environ.get("TEST_DATASET_DIR", "/data/test_dataset")
OUT = os.environ.get("OUT_DIR", "/outputs")
VARIANT = os.environ.get("VARIANT", "aloha-agilex_clean_50")
CAM = os.environ.get("CAMERA", "head_camera")


def main() -> int:
    sys.path.insert(0, os.path.join(FLOWWAM_REPO, "inference"))
    from robot_only_renderer import RobotOnlyRenderer

    eps = sorted(glob.glob(os.path.join(DATA, "data", "fixed_scene_task", "episode*.hdf5")))
    if not eps:
        print(f"FAIL: no episodes under {DATA}", file=sys.stderr)
        return 1
    ep = eps[0]
    name = os.path.splitext(os.path.basename(ep))[0]

    r = RobotOnlyRenderer(embodiment_dir=EMB, variant=VARIANT, render_resolution=(640, 480))
    T = r.get_episode_length(ep)
    n = 6
    idx = [round(i * (T - 1) / (n - 1)) for i in range(n)]
    print(f"episode={name} T={T} render_idx={idx}")

    t0 = time.time()
    frames = r.render_episode(ep, idx, camera=CAM)
    dt = time.time() - t0
    f0 = frames[0]
    print(f"rendered {len(frames)} frames {f0.shape} {f0.dtype} in {dt:.2f}s "
          f"({dt / len(frames) * 1000:.0f} ms/frame)")

    assert f0.shape[:2] == (480, 640), f"unexpected render shape {f0.shape}"
    assert np.isfinite(f0).all(), "non-finite render"
    # motion sanity: first vs last frame should differ (arm moved through the trajectory)
    diff = float(np.mean(np.abs(frames[0].astype(np.int16) - frames[-1].astype(np.int16))))
    nonbg = float(np.mean(np.max(np.abs(f0.astype(np.int16) - f0[0, 0].astype(np.int16)), axis=-1) > 10))
    print(f"frame0-vs-last mean|delta|={diff:.2f}  robot-pixel fraction={nonbg:.3f}")

    os.makedirs(OUT, exist_ok=True)
    strip = np.concatenate(list(frames), axis=1)
    Image.fromarray(strip).save(os.path.join(OUT, "p5_robot_only_strip.png"))
    print(f"saved {OUT}/p5_robot_only_strip.png")

    if diff < 1.0:
        print("FAIL: frames barely change; qpos replay likely not applied", file=sys.stderr)
        return 1
    print("PASS: SAPIEN RobotOnlyRenderer works on ROCm/Vulkan")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
