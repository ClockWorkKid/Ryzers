#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Model-free sanity rollout for the RoboCasa simulator base. Fetches the kitchen assets
# (first run only, ~10GB into the mounted volume), loads the built-in RandomPolicy (no
# weights), rolls it through one kitchen scene on the iGPU (headless EGL), and saves an
# MP4 of the composed 3-view to /outputs. Proves the ROCm/EGL render + MuJoCo step + video
# encode path work end-to-end with no model.
#   ryzers run /ryzers/demos/demo_sim_sanity.sh
#   TASK=OpenDrawer STEPS=120 ryzers run /ryzers/demos/demo_sim_sanity.sh
set -euo pipefail
bash /ryzers/scripts/setup_robocasa.sh
export TASK="${TASK:-TurnOnSinkFaucet}"
export SEED="${SEED:-0}"
export STEPS="${STEPS:-80}"
export OUT_DIR="${OUT_DIR:-/sim_outputs}"
exec python -m sim_robocasa.sanity
