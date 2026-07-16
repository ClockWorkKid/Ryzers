#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Full-model smoke: fetch the released Cosmos3-Nano-Policy-DROID checkpoint and run one
# end-to-end action prediction on ROCm. GATED weights: export HF_TOKEN + accept the license.
#   HF_TOKEN=... ryzers run /ryzers/demos/demo_smoke.sh
set -euo pipefail
bash /ryzers/scripts/download_checkpoints.sh
exec python /ryzers/scripts/model_smoke.py
