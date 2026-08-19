# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Environment sign-of-life for the slim X-WAM policy image on Strix Halo (gfx1151).

Runs inside the built image with NO model weights. Proves the container has (1) a working
ROCm torch on the iGPU and (2) the X-WAM repo + its runtime deps import cleanly on that torch
(exercising the torch-SDPA attention fallback, since flash-attn has no gfx1151 wheel), before
we pull the multi-GB Wan2.2 base + X-WAM checkpoints. This image ships no simulator; the
RoboTwin/RoboCasa stacks live in the simulation/* base images this policy chains onto.
Exits non-zero on any failure so `ryzers run` / CI catches a broken image early.
"""
import os
import sys

XWAM_REPO = os.environ.get("XWAM_REPO", "/repos/xwam")


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

    if XWAM_REPO not in sys.path:
        sys.path.insert(0, XWAM_REPO)

    # X-WAM import graph + core runtime deps.
    import modules.attention          # noqa: F401  (flash-attn absent -> SDPA fallback)
    import modules.t5                  # noqa: F401
    import modules.tokenizers         # noqa: F401
    import modules.vae2_2             # noqa: F401
    import modules.wan_model          # noqa: F401
    from runners.xwam_runner import XWAMRunner  # noqa: F401
    import transformers
    import diffusers                   # noqa: F401
    import lightning                   # noqa: F401
    import einops                      # noqa: F401
    import omegaconf                   # noqa: F401

    from modules.attention import HAS_FLASH_ATTN, attention

    # Exercise the attention op that the DiT relies on, on-device, in the SDPA path.
    q = torch.randn(1, 64, 8, 64, device="cuda", dtype=torch.bfloat16)
    out = attention(q, q, q)
    assert out.shape == q.shape and torch.isfinite(out).all(), "attention fallback produced bad output"

    print(f"transformers     : {transformers.__version__}")
    print(f"diffusers        : {diffusers.__version__}")
    print(f"flash_attn       : {'present' if HAS_FLASH_ATTN else 'absent -> torch SDPA fallback'}")
    print("deps import ok   : modules.{attention,t5,tokenizers,vae2_2,wan_model}, XWAMRunner, "
          "diffusers, lightning, einops, omegaconf")
    print("PASS: X-WAM ROCm policy env OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
