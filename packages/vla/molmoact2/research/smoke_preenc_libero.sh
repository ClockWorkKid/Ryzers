#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Single closed-loop LIBERO smoke for the stage-D pre-encoder pruner. Runs the
# fast path (depth off, num_steps=4, bf16) with VIS_PREENC_KEEP_FRAC and reports
# pc_success. Runs INSIDE the molmoact2 container (call via ryzer_shell.sh).
#   KEEP=0.25 SUITE=libero_object N_EP=1 bash /scripts/smoke_preenc_libero.sh
set -uo pipefail

KEEP="${KEEP:-0.25}"
SUITE="${SUITE:-libero_object}"
N_EP="${N_EP:-1}"
NUM_STEPS="${NUM_STEPS:-4}"
SEED="${SEED:-1000}"
CKPT="${CKPT:-allenai/MolmoAct2-Think-LIBERO}"
OUT="/ryzers/outputs/preenc_smoke"
mkdir -p "$OUT"

export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 TORCH_BLAS_PREFER_HIPBLASLT=0
export MOLMOACT2_DTYPE=bfloat16
export VIS_PREENC_KEEP_FRAC="$KEEP"
export VIS_PREENC_DEBUG=1

LEROBOT_EVAL=lerobot-eval
[ -x /opt/libero-venv/bin/lerobot-eval ] && LEROBOT_EVAL=/opt/libero-venv/bin/lerobot-eval

echo "########## PREENC SMOKE keep=$KEEP suite=$SUITE n_ep=$N_EP $(date '+%F %T') ##########"
"$LEROBOT_EVAL" \
  --policy.type=molmoact2 \
  --policy.checkpoint_path="$CKPT" \
  --policy.inference_action_mode=continuous \
  --policy.enable_depth_reasoning=False \
  --policy.enable_adaptive_depth=False \
  --policy.enable_cuda_graph=False \
  --policy.num_steps="$NUM_STEPS" \
  --policy.norm_tag=libero \
  --policy.device=cuda \
  --env.type=libero \
  --env.task="$SUITE" \
  --eval.batch_size=1 \
  --eval.n_episodes="$N_EP" \
  --seed="$SEED" \
  --output_dir="$OUT/run_${SUITE}_${KEEP}" 2>&1 | tee "$OUT/smoke_${SUITE}_${KEEP}.log"

echo "########## PREENC SMOKE DONE $(date '+%F %T') ##########"
grep -E "pc_success|n_episodes|eval_s" "$OUT/smoke_${SUITE}_${KEEP}.log" | tail -n 5
