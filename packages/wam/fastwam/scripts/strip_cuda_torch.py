# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Strip the CUDA torch stack pins from FastWAM's pyproject so `pip install -e .`
cannot pull cu128 wheels over the base image's ROCm torch build.

Removes any dependency line for torch / torchvision / torchcodec. Everything else
(the exact upstream pins) is preserved so the direct port stays faithful.
"""
import re
import sys
from pathlib import Path

STRIP = ("torch", "torchvision", "torchcodec")


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
