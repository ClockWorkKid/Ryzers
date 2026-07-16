# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Environment sign-of-life for the slim VERA policy image on Strix Halo (gfx1151).

Runs inside the built image with NO model weights. Proves the container has (1) a working
ROCm torch on the iGPU and (2) the VERA package + its two-stage runtime deps (video planner
+ Jacobian IDM) import cleanly, before we pull the multi-GB Wan2.1 bases + checkpoints.
Exits non-zero on any failure so `ryzers run` / CI catches a broken image early.
"""
import sys


def main() -> int:
    import torch

    print(f"torch            : {torch.__version__}")
    print(f"torch.version.hip: {torch.version.hip}")
    if not torch.version.hip:
        print("FAIL: torch is not a ROCm build.", file=sys.stderr)
        return 1
    if not torch.cuda.is_available():
        print("FAIL: no ROCm device visible. Check --device=/dev/kfd, /dev/dri.", file=sys.stderr)
        return 1

    print(f"device[0]        : {torch.cuda.get_device_name(0)}")
    a = torch.randn(512, 512, device="cuda")
    b = torch.randn(512, 512, device="cuda")
    print(f"matmul ok        : sum={(a @ b).sum().item():.3f}")

    # VERA package (both stages) + core runtime deps.
    import vera            # noqa: F401
    import vera.policy     # noqa: F401  closed-loop policy wrappers
    import vera.idm        # noqa: F401  Jacobian inverse-dynamics model (VGGT backbone)
    import vera.server     # noqa: F401  websocket policy server + live viewer
    import vera.video_model  # noqa: F401  WAN + DFoT video planners

    import transformers
    import diffusers
    import hydra           # noqa: F401
    import omegaconf       # noqa: F401
    import einops          # noqa: F401
    import vggt            # noqa: F401  IDM visual backbone

    print(f"transformers     : {transformers.__version__}")
    print(f"diffusers        : {diffusers.__version__}")
    print("deps import ok   : vera{,.policy,.idm,.server,.video_model}, vggt, hydra, diffusers")
    print("PASS: VERA ROCm policy env OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
