#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Per-component runtime latency profile of Cosmos3-Nano-Policy-DROID on gfx1151. Loads the released
# policy, warms up, profiles one steady action inference with CUDA-event forward hooks on every
# module, and writes /outputs/cosmos3_latency_profile.json. Render the diagram with
# scripts/cosmos3_render_arch.py. GATED: export HF_TOKEN + accept the license.
#   HF_TOKEN=... ryzers run /ryzers/demos/demo_latency.sh
set -euo pipefail
bash /ryzers/scripts/download_checkpoints.sh
python /ryzers/scripts/cosmos3_latency_profile.py
exec python /ryzers/scripts/cosmos3_render_arch.py \
  /outputs/cosmos3_latency_profile.json /outputs/cosmos3_arch_latency.png
