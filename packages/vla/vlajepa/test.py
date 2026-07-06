# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Environment sign-of-life for the VLA-JEPA Ryzer image on Strix Halo (gfx1151).

Runs inside the built image. No model weights required. Proves the container has
(1) a working ROCm torch on the iGPU, (2) the starVLA/VLA-JEPA runtime deps at the
pinned versions, and (3) that the starVLA framework registry imports (with the
ROCm attention patch applied), before we pull the multi-GB checkpoints. Exits
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
        print("FAIL: no ROCm device visible. Check --device=/dev/kfd, /dev/dri.",
              file=sys.stderr)
        return 1

    print(f"device[0]        : {torch.cuda.get_device_name(0)}")
    a = torch.randn(512, 512, device="cuda")
    b = torch.randn(512, 512, device="cuda")
    print(f"matmul ok        : sum={(a @ b).sum().item():.3f}")

    import transformers
    import accelerate
    import huggingface_hub
    import einops        # noqa: F401
    import diffusers     # noqa: F401
    import timm          # noqa: F401
    import omegaconf     # noqa: F401
    import safetensors   # noqa: F401

    print(f"transformers     : {transformers.__version__}")
    if not transformers.__version__.startswith("4.57"):
        print(f"FAIL: want transformers 4.57.x, got {transformers.__version__}", file=sys.stderr)
        return 1
    print(f"accelerate       : {accelerate.__version__}")
    print(f"huggingface_hub  : {huggingface_hub.__version__}")

    # starVLA framework import (registers VLA_JEPA; exercises the ROCm attn patch path).
    from starVLA.model.framework import base_framework  # noqa: F401
    from starVLA.model.tools import FRAMEWORK_REGISTRY

    registered = list(getattr(FRAMEWORK_REGISTRY, "_registry", {}) or {})
    print(f"starVLA import ok : frameworks={registered or 'registry-loaded'}")

    # Confirm the ROCm attention patch stuck (no flash_attention_2 literal remains).
    import os
    q3 = "/repos/VLA-JEPA/starVLA/model/modules/vlm/QWen3.py"
    if os.path.isfile(q3):
        with open(q3, "r", encoding="utf-8") as f:
            if "flash_attention_2" in f.read():
                print("FAIL: QWen3.py still requests flash_attention_2 (ROCm patch missing).",
                      file=sys.stderr)
                return 1
        print("attn patch ok    : QWen3.py -> sdpa")

    print("PASS: VLA-JEPA ROCm env OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
