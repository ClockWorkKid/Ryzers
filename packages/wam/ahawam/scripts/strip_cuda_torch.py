# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Prepare AHA-WAM's pyproject for a ROCm `pip install -e .` on the base image.

Two ROCm/porting adjustments, kept minimal so the direct port stays faithful:
  1. Remove any dependency line for torch / torchvision / torchcodec so pip
     cannot pull cu128 wheels over the base image's ROCm torch build.
  2. Relax `requires-python` (upstream pins ">=3.10,<3.11"); the ROCm base image
     ships a newer Python (3.12), and the `ahawam` sources import cleanly there.
     We widen the upper bound rather than dropping the constraint entirely.

Everything else (the exact upstream pins) is preserved.
"""
import re
import sys
from pathlib import Path

STRIP = ("torch", "torchvision", "torchcodec")


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "pyproject.toml")
    dep_pat = re.compile(r'^\s*"(' + "|".join(STRIP) + r')\s*[=<>!~]')
    pyver_pat = re.compile(r'^\s*requires-python\s*=')
    kept, removed, relaxed = [], [], []
    for line in path.read_text().splitlines():
        if dep_pat.match(line):
            removed.append(line.strip())
            continue
        if pyver_pat.match(line):
            relaxed.append(line.strip())
            kept.append('requires-python = ">=3.10"')
            continue
        kept.append(line)
    path.write_text("\n".join(kept) + "\n")
    print("stripped CUDA torch pins:", removed or "(none found)")
    print("relaxed requires-python :", relaxed or "(none found)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
