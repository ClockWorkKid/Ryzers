#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Real-time RoboCasa demo served over HTTP/MJPEG: the sim runs at wall-clock speed while
# the policy plans, so you SEE the planning latency (robot pauses THINKING, then resumes
# as the action buffer refills). Defaults to the built-in RandomPolicy; a policy image
# chains on top and sets POLICY_FACTORY. View at http://localhost:PORT.
#   ryzers run /ryzers/demos/demo_interactive_rt.sh
#   POLICY_FACTORY=xwam_robocasa_policy:build_policy RT_HZ=20 ryzers run /ryzers/demos/demo_interactive_rt.sh
set -euo pipefail
bash /ryzers/scripts/setup_robocasa.sh
export TASK="${TASK:-TurnOnSinkFaucet}"
export SEED="${SEED:-0}"
export PORT="${PORT:-8083}"
export RT_HZ="${RT_HZ:-20}"
export OUT_DIR="${OUT_DIR:-/sim_outputs}"
exec python -m sim_robocasa.interactive_server_rt
