#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# World-model rollout video: decode the model's imagined concat view and render a two-column
# GT|pred clip. On gfx1151/ROCm 7.2.2 the Wan2.2 conv3d decode HANGS in bf16/fp16 -- use fp32
# (slow, offline) via MIOPEN_FIND_MODE=2, or the tiny Conv2D decoder path (see docs/RUNTIME_ANALYSIS.md).
# GATED: export HF_TOKEN + accept the license.
#   HF_TOKEN=... EP=0 DECODE_DTYPE=fp32 MIOPEN_FIND_MODE=2 ryzers run /ryzers/demos/demo_videogen.sh
set -euo pipefail
bash /ryzers/scripts/download_checkpoints.sh
bash /ryzers/scripts/download_datasets.sh
exec python /ryzers/scripts/cosmos3_videogen.py
