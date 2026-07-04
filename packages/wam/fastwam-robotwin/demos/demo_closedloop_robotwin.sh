#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Closed-loop RoboTwin 2.0 rollouts (SAPIEN offscreen Vulkan RT) driven by FastWAM.
# On first run scripts/setup_robotwin.sh fetches the sim assets. Per-task rollout
# videos + results are written under /outputs by the RoboTwin harness.
#   ryzers run /ryzers/demos/demo_closedloop_robotwin.sh
#   TASKS="click_bell lift_pot" NUM_EPISODES=10 ryzers run /ryzers/demos/demo_closedloop_robotwin.sh
set -uo pipefail
TASKS="${TASKS:-beat_block_hammer click_bell place_object_basket handover_block lift_pot}"
TASK_CONFIG="${TASK_CONFIG:-demo_clean}"
NUM_EPISODES="${NUM_EPISODES:-10}"
REL=/models/fastwam_release
export PYTHONPATH="/repos/fastwam/src:/repos/fastwam:/repos/fastwam/experiments/robotwin:/repos/fastwam/third_party/RoboTwin:${PYTHONPATH:-}"

CKPT=$REL/robotwin_uncond_3cam_384.pt
[ -f "$CKPT" ] || { echo "missing $CKPT -> run scripts/download_checkpoints.sh robotwin" >&2; exit 1; }

bash /ryzers/scripts/setup_robotwin.sh

cd /repos/fastwam
for TASK in $TASKS; do
  echo "########## RoboTwin $TASK ($NUM_EPISODES episodes) ##########"
  python -u experiments/robotwin/eval_robotwin_single.py \
    ckpt=$CKPT gpu_id=0 mixed_precision=bf16 \
    EVALUATION.task_name=$TASK EVALUATION.task_config=$TASK_CONFIG \
    EVALUATION.eval_num_episodes=$NUM_EPISODES \
    EVALUATION.dataset_stats_path=$REL/robotwin_uncond_3cam_384_dataset_stats.json \
    2>&1 | grep -viE 'svulkan2|Failed to initialize denoiser|cudaErrorInsufficientDriver' \
    || echo "TASK $TASK returned nonzero"
done
echo "PASS: RoboTwin closed-loop suite complete"
