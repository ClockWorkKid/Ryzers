#!/usr/bin/env python3
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Minimal, idempotent ROCm/gfx1151 source patches for flow-diffusion/AVDC_experiments (rule 2.1:
# rely on upstream, patch only what ROCm / portability needs). String-replacement based so it
# survives small upstream line drift. Run against the checkout: `python rocm_port.py /repos/avdc`.
#
# What it changes and why:
#  1. flowdiffusion/flowdiffusion/goal_diffusion.py - the module calls NVIDIA NVML via pynvml
#     (`print_gpu_utilization()` -> nvmlInit / nvmlDeviceGetMemoryInfo) in the train/validation
#     loop. NVML is CUDA-only and raises on ROCm. Insert an early return so the helper is a no-op
#     (the sampling/inference path never needs it). Idempotent.
import sys, os

# (1) neuter the NVML GPU-utilization helper (CUDA-only) --------------------------------------
NVML_OLD = "def print_gpu_utilization():\n    nvmlInit()"
NVML_NEW = "def print_gpu_utilization():\n    return  # ROCm port: NVIDIA NVML (pynvml) is unavailable on ROCm\n    nvmlInit()"


def replace_in_file(path: str, old: str, new: str, tag: str, count: int = 0) -> bool:
    if not os.path.isfile(path):
        print(f"  [WARN] missing: {path}")
        return False
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    if new in src and old not in src:
        print(f"  [skip] already patched ({tag}): {path}")
        return True
    if old not in src:
        print(f"  [WARN] target not found ({tag}, upstream drift?): {path}")
        return False
    src = src.replace(old, new) if count == 0 else src.replace(old, new, count)
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    print(f"  [ok] patched {tag}: {path}")
    return True


def main() -> int:
    root = sys.argv[1] if len(sys.argv) > 1 else "/repos/avdc"
    fd = os.path.join(root, "flowdiffusion", "flowdiffusion")
    replace_in_file(os.path.join(fd, "goal_diffusion.py"), NVML_OLD, NVML_NEW, "nvml-noop")
    print("rocm_port.py: done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
