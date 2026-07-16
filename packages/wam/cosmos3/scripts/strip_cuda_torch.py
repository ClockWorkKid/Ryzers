# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Strip the CUDA torch stack + NVIDIA-only kernel pins from cosmos-framework's pyproject so
the ROCm install cannot pull cu13x wheels over the base image's ROCm torch, nor force a numpy
downgrade, nor require NVIDIA-only attention/kernels that have no gfx1151 build.

Removes any dependency line for the listed packages (matched in both `[project].dependencies`
and `[dependency-groups]` tables). The base image's torch/numpy are then held via the
PIP_CONSTRAINT pin in the Dockerfile; NVIDIA attention/kernels are replaced by SDPA/AOTriton
fallbacks (patches/, Phase 1). Everything else (the exact upstream pins) is preserved so the
direct port stays faithful.
"""
import re
import sys
from pathlib import Path

STRIP = (
    "torch", "torchvision", "torchcodec", "numpy",
    "transformer-engine", "transformer_engine",
    "natten", "flash-attn", "flash_attn", "flash-attn-3-nv", "apex", "apex-nv",
)


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "pyproject.toml")
    names = "|".join(re.escape(s) for s in STRIP)
    pat = re.compile(r'^\s*"(' + names + r')\s*[=<>!~;\[]')
    kept, removed = [], []
    for line in path.read_text().splitlines():
        if pat.match(line):
            removed.append(line.strip())
        else:
            kept.append(line)
    path.write_text("\n".join(kept) + "\n")
    print("stripped CUDA/NVIDIA-only pins:")
    for r in removed:
        print("  -", r)
    if not removed:
        print("  (none found — verify the pyproject layout / update STRIP)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
