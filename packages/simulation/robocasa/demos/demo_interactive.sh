#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Interactive RoboCasa demo (chunk-replay) served over HTTP/MJPEG. Defaults to the
# built-in RandomPolicy; a policy image chains on top and sets POLICY_FACTORY to drive it
# with a real model. View at http://localhost:PORT (ssh -L PORT:localhost:PORT <host>).
#   ryzers run /ryzers/demos/demo_interactive.sh
#   POLICY_FACTORY=xwam_robocasa_policy:build_policy ryzers run /ryzers/demos/demo_interactive.sh
set -euo pipefail
bash /ryzers/scripts/setup_robocasa.sh
export TASK="${TASK:-TurnOnSinkFaucet}"
export SEED="${SEED:-0}"
export PORT="${PORT:-8082}"
export OUT_DIR="${OUT_DIR:-/sim_outputs}"
exec python -m sim_robocasa.interactive_server
