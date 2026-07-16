# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Strip the CUDA torch stack + numpy pins from VERA's pyproject so `pip install -e .`
cannot pull cu124 wheels over the base image's ROCm torch build, nor force a numpy
downgrade that conflicts with the base image's numpy.

VERA pins `torch==2.6.0` (ABI-tied to flash-attn CUDA wheels). On ROCm we keep the base
image's torch 2.10 build instead, held via the PIP_CONSTRAINT pin in the Dockerfile.
Removes any core/extra dependency line for torch / torchvision / torchcodec / numpy;
everything else (the exact upstream pins) is preserved so the direct port stays faithful.
"""
import re
import sys
from pathlib import Path

STRIP = ("torch", "torchvision", "torchcodec", "numpy")


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "pyproject.toml")
    pat = re.compile(r'^\s*"(' + "|".join(STRIP) + r')\s*[=<>!~]')
    kept, removed = [], []
    for line in path.read_text().splitlines():
        if pat.match(line):
            removed.append(line.strip())
        else:
            kept.append(line)
    path.write_text("\n".join(kept) + "\n")
    print("stripped CUDA torch pins:", removed or "(none found)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
