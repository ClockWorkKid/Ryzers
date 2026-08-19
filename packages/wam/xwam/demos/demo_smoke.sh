#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Full-model smoke: load a released X-WAM SFT checkpoint and run one end-to-end action
# prediction (the fast async-denoising action path) on ROCm. First run downloads the Wan2.2
# base weights + the SFT checkpoint into the mounted cache. Usage:
#   EXP=robotwin_sft ryzers run --name xwam /ryzers/demos/demo_smoke.sh
set -euo pipefail

EXP="${EXP:-robotwin_sft}"
bash /ryzers/scripts/download_checkpoints.sh "${EXP%_sft}"

exec python /ryzers/scripts/model_smoke.py
