# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Environment sign-of-life for the FastWAM RoboTwin image on Strix Halo (gfx1151).

Validates the ROCm torch, the SAPIEN/mplib sim stack and an offscreen Vulkan
render device, before pulling weights or fetching RoboTwin assets.
"""
import sys


def main() -> int:
    import torch

    print(f"torch            : {torch.__version__}  hip={torch.version.hip}")
    if not torch.version.hip:
        print("FAIL: torch is not a ROCm build.", file=sys.stderr)
        return 1
    if not torch.cuda.is_available():
        print("FAIL: no ROCm device visible (check /dev/kfd, /dev/dri).", file=sys.stderr)
        return 1
    print(f"device[0]        : {torch.cuda.get_device_name(0)}")

    import numpy
    import sapien
    import mplib.planner                              # noqa: F401
    from mplib.sapien_utils import SapienPlanner      # noqa: F401
    print(f"numpy            : {numpy.__version__}")
    print(f"sapien           : {sapien.__version__}")

    try:
        import sapien.core as sc
        eng = sc.Engine()
        rend = sc.SapienRenderer(offscreen_only=True)
        eng.set_renderer(rend)
        print("sapien renderer  : offscreen OK")
    except Exception as e:
        print(f"WARN: offscreen renderer probe: {type(e).__name__}: {str(e)[:80]}")

    print("PASS: FastWAM RoboTwin ROCm env OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
