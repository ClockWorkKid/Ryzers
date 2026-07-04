#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Full-model smoke: load the released FastWAM LIBERO checkpoint and run one
# end-to-end action prediction on ROCm. First run downloads the Wan2.2 base
# weights + checkpoint into the mounted cache. Usage:
#   ryzers run /ryzers/demos/demo_smoke.sh
set -euo pipefail

# Ensure the checkpoint is present (idempotent; resumes from cache).
bash /ryzers/scripts/download_checkpoints.sh libero

exec python /ryzers/scripts/model_smoke.py
