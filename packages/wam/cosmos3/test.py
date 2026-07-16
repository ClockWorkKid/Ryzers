# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Environment sign-of-life for the slim Cosmos3-Nano-Policy image on Strix Halo (gfx1151).

Runs inside the built image with NO model weights. Proves the container has (1) a working
ROCm torch on the iGPU and (2) the cosmos-framework package + its runtime deps import cleanly,
before pulling the ~32 GB+ Cosmos3-Nano-Policy-DROID checkpoint. Ships no simulator. Exits
non-zero on any failure so `ryzers run` / CI catches a broken image early.
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

    # cosmos-framework package + core runtime deps.
    import cosmos_framework            # noqa: F401
    import transformers
    import diffusers                   # noqa: F401
    import safetensors                 # noqa: F401
    import einops                      # noqa: F401
    import omegaconf                   # noqa: F401

    print(f"transformers     : {transformers.__version__}")
    print(f"diffusers        : {diffusers.__version__}")
    print("deps import ok   : cosmos_framework, transformers, diffusers, safetensors, einops, omegaconf")
    print("PASS: Cosmos3 ROCm policy env OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
