# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Strip base-image-owned / CUDA-only dependency pins from FlowWAM's requirements.txt so
`pip install -r requirements.txt && pip install -e .` cannot (a) pull CUDA wheels over the
base image's ROCm torch, (b) downgrade the base's numpy, or (c) reinstall the base's SAPIEN
(the robotwin sim base already ships a working Vulkan SAPIEN on gfx1151).

Removed lines are held instead via the PIP_CONSTRAINT pin in the Dockerfile (see the
"Pin the base's torch + numpy + sapien" note there). This lets the FlowWAM (diffsynth) layer
compose faithfully on top of the robotwin sim base. Everything else (the exact upstream pins)
is preserved so the direct port stays faithful.

FlowWAM's requirements.txt does NOT list flash_attn / apex / nvidia-cublas (those are
README-only manual CUDA steps we deliberately skip: DiffSynth's flash_attention() already
falls back to torch SDPA on ROCm, and apex is only needed by the deferred SeedVR refiner).
We still strip them defensively in case a future pin appears.
"""
import re
import sys
from pathlib import Path

STRIP = (
    "torch", "torchvision", "torchcodec", "torchaudio", "numpy", "sapien",
    "flash_attn", "flash-attn", "apex", "nvidia-cublas-cu12", "sageattention",
)


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "requirements.txt")
    names = "|".join(re.escape(s) for s in STRIP)
    pat = re.compile(r'^\s*(' + names + r')\s*([=<>!~\[]|$)', re.IGNORECASE)
    kept, removed = [], []
    for line in path.read_text().splitlines():
        if pat.match(line.strip()):
            removed.append(line.strip())
        else:
            kept.append(line)
    path.write_text("\n".join(kept) + "\n")
    print("stripped base-owned/CUDA pins:", removed or "(none found)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
