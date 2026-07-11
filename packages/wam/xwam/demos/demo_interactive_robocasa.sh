#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# X-WAM interactive RoboCasa demo (chunk-replay) over HTTP/MJPEG, on the simulation/robocasa
# base. Requires the chain:  ryzers build robocasa xwam. Command-driven: pick a kitchen task,
# press Run, watch X-WAM roll it out live. View at http://localhost:PORT.
#   ryzers run /ryzers/demos/demo_interactive_robocasa.sh
#   TASK=OpenDrawer ryzers run /ryzers/demos/demo_interactive_robocasa.sh
set -uo pipefail

if [ ! -d /opt/robocasa ]; then
  echo "ERROR: simulation/robocasa base not found. Build the chain: ryzers build robocasa xwam" >&2
  exit 1
fi

export TASK="${TASK:-TurnOnSinkFaucet}"
export SEED="${SEED:-0}"
export PORT="${PORT:-8082}"
export EXP="${EXP:-robocasa_sft}"
export DENOISE_STEPS="${DENOISE_STEPS:-50}"
export ACTION_DENOISE_STEPS="${ACTION_DENOISE_STEPS:-10}"
export ACTION_LENGTH="${ACTION_LENGTH:-32}"
export REPLAN_STEPS="${REPLAN_STEPS:-${ACTION_LENGTH}}"
export CFG="${CFG:-0.0}"

bash /ryzers/scripts/download_checkpoints.sh robocasa
bash /ryzers/scripts/setup_robocasa.sh

# /opt/sim holds the sim_robocasa harness (base ENV PYTHONPATH, clobbered by this layer's
# ENV PYTHONPATH=/repos/xwam) -- re-add it explicitly.
export PYTHONPATH="/ryzers/experiments/robocasa_xwam/xwam_policy:/ryzers/experiments:/opt/sim:${PYTHONPATH:-}"
export POLICY_FACTORY="deploy_policy:build_policy"
exec python -u -m sim_robocasa.interactive_server
