#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Closed-loop RoboTwin 2.0 rollouts (SAPIEN offscreen Vulkan RT) driven by AHA-WAM, on the
# de-vendored simulation/robotwin base. Requires the chain:  ryzers build robotwin ahawam.
# RoboTwin's own script/eval_policy.py is the model-agnostic runner; the upstream
# experiments/robotwin/eval_robotwin_single.py symlinks the AHA-WAM ahawam_policy plugin
# into /opt/RoboTwin/policy and runs it (async video-prefill + action-chunk schedule via
# chunks_per_video_prefill). Per-task rollout videos + results land under /outputs.
#   ryzers run /ryzers/demos/demo_closedloop_robotwin.sh
#   TASKS="click_bell lift_pot" NUM_EPISODES=10 ryzers run /ryzers/demos/demo_closedloop_robotwin.sh
set -uo pipefail
ulimit -n 65536 2>/dev/null || true  # SAPIEN/RoboTwin open many fds; raise soft nofile (hard=524288)
if [ ! -d /opt/RoboTwin ]; then
  echo "ERROR: simulation/robotwin base not found (no /opt/RoboTwin)." >&2
  echo "       Build the chain:  ryzers build robotwin ahawam" >&2
  exit 1
fi
TASKS="${TASKS:-beat_block_hammer click_bell place_object_basket handover_block lift_pot}"
TASK_CONFIG="${TASK_CONFIG:-demo_clean}"
NUM_EPISODES="${NUM_EPISODES:-10}"
CHUNKS_PER_VIDEO_PREFILL="${CHUNKS_PER_VIDEO_PREFILL:-2}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-10}"
REL=/models/ahawam_release
export PYTHONPATH="/repos/ahawam/src:/repos/ahawam:/repos/ahawam/experiments/robotwin:/opt/RoboTwin:/opt/sim:${PYTHONPATH:-}"
export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:True}"

CKPT="${CKPT:-$REL/robotwin_ahawam.pt}"
STATS="${DATASET_STATS:-$REL/dataset_stats.json}"

# Fetch the checkpoint if missing (idempotent; resumes from cache).
bash /ryzers/scripts/download_checkpoints.sh robotwin
[ -f "$CKPT" ] || { echo "missing $CKPT -> run scripts/download_checkpoints.sh robotwin" >&2; exit 1; }

# Fetch RoboTwin assets + wire them into /opt/RoboTwin (idempotent; sim base script).
bash /ryzers/scripts/setup_robotwin.sh

cd /repos/ahawam
for TASK in $TASKS; do
  echo "########## RoboTwin $TASK ($NUM_EPISODES episodes) ##########"
  python -u experiments/robotwin/eval_robotwin_single.py \
    ckpt=$CKPT gpu_id=0 mixed_precision=bf16 \
    EVALUATION.robotwin_root=/opt/RoboTwin \
    EVALUATION.task_name=$TASK EVALUATION.task_config=$TASK_CONFIG \
    EVALUATION.eval_num_episodes=$NUM_EPISODES \
    EVALUATION.chunks_per_video_prefill=$CHUNKS_PER_VIDEO_PREFILL \
    EVALUATION.num_inference_steps=$NUM_INFERENCE_STEPS \
    EVALUATION.dataset_stats_path=$STATS \
    2>&1 | grep -viE 'svulkan2|Failed to initialize denoiser|cudaErrorInsufficientDriver' \
    || echo "TASK $TASK returned nonzero"
  # eval writes to the ephemeral repo dir; copy rollout videos + results to the
  # mounted artifacts volume so they survive the --rm container.
  if [ -d /repos/ahawam/evaluate_results ]; then
    mkdir -p "${OUT_DIR:-/outputs}/robotwin_eval"
    cp -ra /repos/ahawam/evaluate_results/. "${OUT_DIR:-/outputs}/robotwin_eval/" 2>/dev/null || true
  fi
done
echo "PASS: RoboTwin closed-loop suite complete"
