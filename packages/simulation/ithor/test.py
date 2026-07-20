# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""iTHOR simulator base sign-of-life + headless Vulkan render spike (the Phase-2 GATE).

This is the make-or-break check for the port: it proves ai2thor can render offscreen through
Vulkan (platform=CloudRendering) on gfx1151 with no X server. It:
  1. imports ai2thor + the sim_ithor harness,
  2. starts a CloudRendering Controller, resets FloorPlan1, steps a few discrete actions,
  3. asserts a non-blank RGB frame + a valid depth frame come back,
  4. prints the render platform / device.
If Vulkan/RADV cannot bring up the Unity build this raises (see the fallback ladder in the package
README / laptop docs/ithor/PLAN.md: RADV -> AMDVLK -> newer ai2thor build -> Xvfb)."""
import os
import sys

import numpy as np


def _vulkan_summary():
    """Best-effort: print the Vulkan device the loader sees (does not fail the gate)."""
    import shutil
    import subprocess

    exe = shutil.which("vulkaninfo")
    if not exe:
        print("  (vulkaninfo not found; skipping Vulkan enumeration)")
        return
    try:
        out = subprocess.run([exe, "--summary"], capture_output=True, text=True, timeout=60).stdout
        for line in out.splitlines():
            if any(k in line for k in ("deviceName", "driverName", "apiVersion", "GPU id")):
                print("  vulkan:", line.strip())
    except Exception as e:  # noqa: BLE001
        print(f"  (vulkaninfo failed: {e})")


def main() -> int:
    import ai2thor
    from ai2thor.platform import CloudRendering  # noqa: F401
    from sim_ithor import Policy, load_policy  # noqa: F401
    from sim_ithor.tasks import SCENE2TARGETS, get_cmat  # noqa: F401
    from sim_ithor.env import ThorEnv

    print(f"ai2thor {ai2thor.__version__} | numpy {np.__version__} | "
          f"platform={os.environ.get('THOR_PLATFORM', 'CloudRendering')} "
          f"gpu_device={os.environ.get('THOR_GPU_DEVICE', '0')}")
    print(f"VK_ICD_FILENAMES={os.environ.get('VK_ICD_FILENAMES', '(unset)')}")
    _vulkan_summary()

    scene = "FloorPlan1"
    target = SCENE2TARGETS[scene][0]
    print(f"bringing up CloudRendering controller: scene={scene} target={target} ...")
    env = ThorEnv(scene, target, seed=0, resolution=(64, 64), max_eplen=10)
    try:
        frame, depth = env.reset()
        assert frame is not None and frame.shape == (64, 64, 3), f"bad RGB frame: {getattr(frame, 'shape', None)}"
        assert depth is not None and depth.shape == (64, 64), f"bad depth frame: {getattr(depth, 'shape', None)}"
        assert int(np.asarray(frame).sum()) > 0, "RGB frame is all-black (render likely failed)"
        for action in ("RotateRight", "MoveAhead", "RotateLeft"):
            (frame, depth), success, done = env.step(action)
        print(f"  frame={frame.shape} dtype={frame.dtype} sum={int(np.asarray(frame).sum())} | "
              f"depth={depth.shape} range=[{float(depth.min()):.2f},{float(depth.max()):.2f}] m")
    finally:
        env.close()

    print("PASS: iTHOR CloudRendering headless render OK on this device")
    return 0


if __name__ == "__main__":
    sys.exit(main())
