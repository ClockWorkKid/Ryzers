#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# X-WAM REAL-TIME interactive RoboTwin 2.0 demo over HTTP/MJPEG, on the de-vendored
# simulation/robotwin base. Requires the chain:  ryzers build robotwin xwam. The sim runs at
# wall-clock speed while X-WAM plans, so you SEE the planning latency (dual arms HOLD their
# current EE pose = THINKING, then resume as the action buffer refills).
# View at http://localhost:PORT (ssh -L PORT:localhost:PORT <host>).
#   ryzers run /ryzers/demos/demo_interactive_robotwin_rt.sh
#   TASK=beat_block_hammer ryzers run /ryzers/demos/demo_interactive_robotwin_rt.sh
set -uo pipefail

if [ ! -d /opt/RoboTwin ]; then
  echo "ERROR: simulation/robotwin base not found (no /opt/RoboTwin)." >&2
  echo "       Build the chain:  ryzers build robotwin xwam" >&2
  exit 1
fi

export TASK="${TASK:-beat_block_hammer}"
export TASK_CONFIG="${TASK_CONFIG:-demo_clean}"
export SEED="${SEED:-100000}"
export PORT="${PORT:-8083}"
export EXP="${EXP:-robotwin_sft}"
export DENOISE_STEPS="${DENOISE_STEPS:-50}"
export ACTION_DENOISE_STEPS="${ACTION_DENOISE_STEPS:-10}"
export ACTION_LENGTH="${ACTION_LENGTH:-32}"
# Smaller replan window keeps the real-time buffer responsive between forwards.
export REPLAN_STEPS="${REPLAN_STEPS:-8}"
export CFG="${CFG:-0.0}"

bash /ryzers/scripts/download_checkpoints.sh robotwin
bash /ryzers/scripts/setup_robotwin.sh

# The xwam layer's ENV PYTHONPATH=/repos/xwam clobbers the base's /opt/sim:/opt/RoboTwin
# (sim_robotwin harness + RoboTwin envs.<task>); re-add them, plus the adapter + experiments
# (xwam_core) dirs.
export PYTHONPATH="/ryzers/experiments/robotwin_xwam/xwam_policy:/ryzers/experiments:/opt/sim:/opt/RoboTwin:/repos/xwam:${PYTHONPATH:-}"
export POLICY_FACTORY="deploy_policy:build_policy"
exec python -u -m sim_robotwin.interactive_server_rt
