#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# X-WAM interactive RoboTwin 2.0 demo (chunk-replay) over HTTP/MJPEG, on the de-vendored
# simulation/robotwin base. Requires the chain:  ryzers build robotwin xwam. Command-driven:
# pick a task, press Run, watch X-WAM roll it out live (dual-arm EE control via RoboTwin IK).
# View at http://localhost:PORT (ssh -L PORT:localhost:PORT <host>).
#   ryzers run /ryzers/demos/demo_interactive_robotwin.sh
#   TASK=beat_block_hammer ryzers run /ryzers/demos/demo_interactive_robotwin.sh
set -uo pipefail

if [ ! -d /opt/RoboTwin ]; then
  echo "ERROR: simulation/robotwin base not found (no /opt/RoboTwin)." >&2
  echo "       Build the chain:  ryzers build robotwin xwam" >&2
  exit 1
fi

export TASK="${TASK:-beat_block_hammer}"
export TASK_CONFIG="${TASK_CONFIG:-demo_clean}"
export SEED="${SEED:-100000}"
export PORT="${PORT:-8082}"
export EXP="${EXP:-robotwin_sft}"
export DENOISE_STEPS="${DENOISE_STEPS:-50}"
export ACTION_DENOISE_STEPS="${ACTION_DENOISE_STEPS:-10}"
export ACTION_LENGTH="${ACTION_LENGTH:-32}"
# Sync demo replays whole chunks; execute the full predicted chunk before replanning.
export REPLAN_STEPS="${REPLAN_STEPS:-${ACTION_LENGTH}}"
export CFG="${CFG:-0.0}"

bash /ryzers/scripts/download_checkpoints.sh robotwin
bash /ryzers/scripts/setup_robotwin.sh

# The xwam layer's ENV PYTHONPATH=/repos/xwam clobbers the base's /opt/sim:/opt/RoboTwin
# (sim_robotwin harness + RoboTwin envs.<task>); re-add them, plus the adapter + experiments
# (xwam_core) dirs.
export PYTHONPATH="/ryzers/experiments/robotwin_xwam/xwam_policy:/ryzers/experiments:/opt/sim:/opt/RoboTwin:/repos/xwam:${PYTHONPATH:-}"
export POLICY_FACTORY="deploy_policy:build_policy"
exec python -u -m sim_robotwin.interactive_server
