#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Full-model smoke: load the released FLUX.2 ImageWAM LIBERO checkpoint and run one
# end-to-end action prediction on ROCm (add VISUALIZE_DREAM=1 to also render the dreamed
# edited future frame). First run downloads the ImageWAM checkpoint + FLUX.2 base/AE (gated;
# needs HF_TOKEN) + Qwen3 into the mounted cache. Usage:
#   ryzers run --name imagewam /ryzers/demos/demo_smoke.sh
set -euo pipefail

# Ensure weights are present (idempotent; resumes from cache). Gated FLUX.2 base/AE are
# skipped with a warning if HF_TOKEN is unset.
bash /ryzers/scripts/download_checkpoints.sh "${SUITE:-libero}" "${FLUX2_VARIANT:-4b}"

exec python /ryzers/scripts/model_smoke.py
