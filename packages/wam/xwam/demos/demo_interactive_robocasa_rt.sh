#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# X-WAM REAL-TIME interactive RoboCasa demo over HTTP/MJPEG, on the simulation/robocasa base.
# Requires the chain:  ryzers build robocasa xwam. The sim runs at wall-clock speed while
# X-WAM plans, so you SEE the planning latency (robot HOLDs THINKING, then resumes as the
# action buffer refills). View at http://localhost:PORT.
#   ryzers run /ryzers/demos/demo_interactive_robocasa_rt.sh
#   TASK=OpenDrawer RT_HZ=20 ryzers run /ryzers/demos/demo_interactive_robocasa_rt.sh
set -uo pipefail

if [ ! -d /opt/robocasa ]; then
  echo "ERROR: simulation/robocasa base not found. Build the chain: ryzers build robocasa xwam" >&2
  exit 1
fi

export TASK="${TASK:-TurnOnSinkFaucet}"
export SEED="${SEED:-0}"
export PORT="${PORT:-8083}"
export RT_HZ="${RT_HZ:-20}"
export EXP="${EXP:-robocasa_sft}"
export DENOISE_STEPS="${DENOISE_STEPS:-50}"
export ACTION_DENOISE_STEPS="${ACTION_DENOISE_STEPS:-10}"
export ACTION_LENGTH="${ACTION_LENGTH:-32}"
# Smaller replan window keeps the real-time buffer responsive between forwards.
export REPLAN_STEPS="${REPLAN_STEPS:-8}"
export CFG="${CFG:-0.0}"

bash /ryzers/scripts/download_checkpoints.sh robocasa
bash /ryzers/scripts/setup_robocasa.sh

# /opt/sim holds the sim_robocasa harness (base ENV PYTHONPATH, clobbered by this layer's
# ENV PYTHONPATH=/repos/xwam) -- re-add it explicitly.
export PYTHONPATH="/ryzers/experiments/robocasa_xwam/xwam_policy:/ryzers/experiments:/opt/sim:${PYTHONPATH:-}"
export POLICY_FACTORY="deploy_policy:build_policy"
exec python -u -m sim_robocasa.interactive_server_rt
