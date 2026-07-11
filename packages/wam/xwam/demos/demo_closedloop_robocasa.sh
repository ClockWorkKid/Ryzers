#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# X-WAM closed-loop RoboCasa rollouts (robosuite / MuJoCo, headless EGL) on the new
# simulation/robocasa base. Requires the chain:  ryzers build robocasa xwam.
# Drives the X-WAM sim_robocasa.Policy adapter (direct in-process inference, 7-D delta-EE
# via robosuite OSC_POSE) through the model-agnostic closed-loop runner, writing per-episode
# videos + _result.json under /outputs/robocasa/<task>. Weights + kitchen assets are fetched
# at runtime (rules 3 & 8).
#   ryzers run /ryzers/demos/demo_closedloop_robocasa.sh
#   TASK=TurnOnSinkFaucet NUM_EVALS=1 ryzers run /ryzers/demos/demo_closedloop_robocasa.sh
set -uo pipefail

if [ ! -d /opt/robocasa ]; then
  echo "ERROR: simulation/robocasa base not found (no /opt/robocasa)." >&2
  echo "       Build the chain:  ryzers build robocasa xwam" >&2
  exit 1
fi

export TASK="${TASK:-TurnOnSinkFaucet}"
export NUM_EVALS="${NUM_EVALS:-1}"
export SEED_BASE="${SEED_BASE:-0}"
export MAX_STEPS="${MAX_STEPS:-0}"
export EXP="${EXP:-robocasa_sft}"
export DENOISE_STEPS="${DENOISE_STEPS:-50}"
export ACTION_DENOISE_STEPS="${ACTION_DENOISE_STEPS:-10}"
export ACTION_LENGTH="${ACTION_LENGTH:-32}"
export REPLAN_STEPS="${REPLAN_STEPS:-${ACTION_LENGTH}}"
export CFG="${CFG:-0.0}"

# Fetch X-WAM weights (Wan2.2-5B base + robocasa_sft) if not already cached (idempotent).
bash /ryzers/scripts/download_checkpoints.sh robocasa

# Fetch RoboCasa kitchen assets into the mounted volume (idempotent; sim base script).
bash /ryzers/scripts/setup_robocasa.sh

# Select the X-WAM RoboCasa policy adapter via the sim_robocasa POLICY_FACTORY seam.
# /opt/sim holds the sim_robocasa harness (set by the base's ENV PYTHONPATH, which this
# layer's ENV PYTHONPATH=/repos/xwam clobbers) -- re-add it explicitly here.
export PYTHONPATH="/ryzers/experiments/robocasa_xwam/xwam_policy:/ryzers/experiments:/opt/sim:${PYTHONPATH:-}"
export POLICY_FACTORY="deploy_policy:build_policy"

echo "########## X-WAM RoboCasa $TASK ($NUM_EVALS evals) ##########"
python -u -m sim_robocasa.closedloop
RC=$?

echo "PASS: X-WAM RoboCasa closed-loop complete (rc=$RC)"
exit $RC
