#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# X-WAM closed-loop RoboTwin 2.0 rollouts (SAPIEN offscreen Vulkan RT) on the de-vendored
# simulation/robotwin base. Requires the chain:  ryzers build robotwin xwam.
# RoboTwin's model-agnostic script/eval_policy.py runs the render smoke, drives the X-WAM
# deploy_policy plugin (direct in-process inference, delta-EE control), and writes per-episode
# videos + _result_*.txt. Weights are fetched at runtime (rule 8) into the mounted /models cache.
#   ryzers run /ryzers/demos/demo_closedloop_robotwin.sh
#   TASKS="beat_block_hammer" NUM_EPISODES=1 ryzers run /ryzers/demos/demo_closedloop_robotwin.sh
set -uo pipefail

if [ ! -d /opt/RoboTwin ]; then
  echo "ERROR: simulation/robotwin base not found (no /opt/RoboTwin)." >&2
  echo "       Build the chain:  ryzers build robotwin xwam" >&2
  exit 1
fi

TASKS="${TASKS:-${TASK:-beat_block_hammer}}"
export TASK_CONFIG="${TASK_CONFIG:-demo_clean}"
export NUM_EPISODES="${NUM_EPISODES:-1}"
export EXP="${EXP:-robotwin_sft}"
export DENOISE_STEPS="${DENOISE_STEPS:-50}"
export ACTION_DENOISE_STEPS="${ACTION_DENOISE_STEPS:-10}"
export ACTION_LENGTH="${ACTION_LENGTH:-32}"
export REPLAN_STEPS="${REPLAN_STEPS:-${ACTION_LENGTH}}"
export CFG="${CFG:-0.0}"

# Fetch X-WAM weights (Wan2.2-5B base + robotwin_sft) if not already cached (idempotent).
bash /ryzers/scripts/download_checkpoints.sh robotwin

# Fetch RoboTwin assets + wire them into /opt/RoboTwin (idempotent; sim base script).
bash /ryzers/scripts/setup_robotwin.sh

RC=0
for TASK in $TASKS; do
  echo "########## X-WAM RoboTwin $TASK ($NUM_EPISODES ep, config=$TASK_CONFIG) ##########"
  TASK="$TASK" EVAL_OUTPUT_DIR="/outputs/robotwin/$TASK" \
    python -u /ryzers/experiments/robotwin/eval_robotwin_single.py \
    2>&1 | grep -viE 'svulkan2|Failed to initialize denoiser|cudaErrorInsufficientDriver' \
    || { echo "TASK $TASK returned nonzero"; RC=1; }
done

echo "PASS: X-WAM RoboTwin closed-loop suite complete (rc=$RC)"
exit $RC
