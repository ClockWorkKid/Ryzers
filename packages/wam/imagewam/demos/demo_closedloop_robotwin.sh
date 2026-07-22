#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Closed-loop RoboTwin 2.0 rollouts (SAPIEN offscreen Vulkan) driven by FLUX.2 ImageWAM, on
# the de-vendored simulation/robotwin base. Requires the chain:  ryzers build robotwin imagewam.
# RoboTwin's own script/eval_policy.py is the parity-critical, model-agnostic runner;
# eval_robotwin_single.py symlinks experiments/robotwin/imagewam_policy into /opt/RoboTwin/policy
# and forwards the resolved cfg.model (flux2 paths, variant, proprio_dim, ...) as model_overrides.
#   ryzers run /ryzers/demos/demo_closedloop_robotwin.sh
#   TASKS="click_bell place_object_basket" NUM_EPISODES=10 ryzers run /ryzers/demos/demo_closedloop_robotwin.sh
set -uo pipefail
if [ ! -d /opt/RoboTwin ]; then
  echo "ERROR: simulation/robotwin base not found (no /opt/RoboTwin)." >&2
  echo "       Build the chain:  ryzers build robotwin imagewam" >&2
  exit 1
fi

REPO="${IMAGEWAM_REPO:-/repos/imagewam}"
FLUX2_SRC="${FLUX2_SRC:-/repos/flux2}"
VARIANT="${FLUX2_VARIANT:-4b}"
TASKS="${TASKS:-place_object_basket click_bell beat_block_hammer}"
TASK_CONFIG="${TASK_CONFIG:-demo_clean}"
NUM_EPISODES="${NUM_EPISODES:-10}"
TASK_CFG_NAME="${TASK_CFG_NAME:-robotwin_flux2_klein_${VARIANT}_base_clean_imagewam}"

CKPT="${CKPT_PATH:-/models/imagewam_release/robotwin/flux2_klein_${VARIANT}/model.pt}"
STATS="${DATASET_STATS_PATH:-/models/imagewam_release/robotwin/flux2_klein_${VARIANT}/dataset_stats.json}"
AE="${FLUX2_AE_MODEL_PATH:-/models/flux2/FLUX.2-klein-base-4B/ae.safetensors}"
DIT="${FLUX2_MODEL_PATH:-/models/flux2/FLUX.2-klein-base-4B/flux-2-klein-base-4b.safetensors}"
QWEN3="${FLUX2_QWEN3_MODEL_SPEC:-Qwen/Qwen3-4B}"
for f in "$CKPT" "$STATS" "$AE" "$DIT"; do
  [ -f "$f" ] || { echo "missing weight: $f" >&2; exit 1; }
done

export PYTHONPATH="${REPO}/src:${FLUX2_SRC}/src:${FLUX2_SRC}:${REPO}/experiments/robotwin:/opt/RoboTwin:${PYTHONPATH:-}"

# Fetch RoboTwin assets + wire them into /opt/RoboTwin (idempotent; provided by the sim base).
bash /ryzers/scripts/setup_robotwin.sh

cd "$REPO"
for TASK in $TASKS; do
  echo "########## RoboTwin $TASK ($TASK_CONFIG, $NUM_EPISODES episodes) ##########"
  python -u experiments/robotwin/eval_robotwin_single.py \
    --config-name sim_robotwin \
    task="$TASK_CFG_NAME" \
    ckpt="$CKPT" gpu_id=0 mixed_precision=bf16 \
    EVALUATION.robotwin_root=/opt/RoboTwin \
    EVALUATION.task_name="$TASK" EVALUATION.task_config="$TASK_CONFIG" \
    EVALUATION.eval_num_episodes="$NUM_EPISODES" \
    EVALUATION.action_horizon="${ACTION_HORIZON:-16}" EVALUATION.replan_steps="${REPLAN_STEPS:-16}" \
    EVALUATION.num_inference_steps="${NUM_INFERENCE_STEPS:-10}" \
    EVALUATION.robotwin_camera_layout="${ROBOTWIN_CAMERA_LAYOUT:-compact_288x256}" \
    EVALUATION.skip_get_obs_within_replan=true \
    EVALUATION.dataset_stats_path="$STATS" \
    model.flux2_src_path="$FLUX2_SRC" model.flux2_model_path="$DIT" \
    model.ae_model_path="$AE" model.variant="klein-base-${VARIANT}" \
    model.qwen3_model_spec="$QWEN3" model.load_text_encoder=true \
    model.pack_proprio_after_text=true model.proprio_dim="${PROPRIO_DIM:-14}" \
    2>&1 | grep -aviE 'svulkan2|Failed to initialize denoiser|cudaErrorInsufficientDriver' \
    || echo "TASK $TASK returned nonzero"
done
echo "PASS: RoboTwin closed-loop suite complete (see /opt/RoboTwin eval_result + /outputs logs)"
