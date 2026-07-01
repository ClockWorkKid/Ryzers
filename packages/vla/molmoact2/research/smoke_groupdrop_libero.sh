#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Quick closed-loop smoke for Route-A group-drop on the easiest pick-and-place
# suite (libero_object), 1 episode/task (10 tasks), bf16 fast path. Runs INSIDE
# the molmoact2 container.
set -uo pipefail
OUT=/ryzers/outputs/groupdrop_smoke
mkdir -p "$OUT"
CKPT="${CKPT:-allenai/MolmoAct2-Think-LIBERO}"
SUITE="${SUITE:-libero_object}"
FRACS="${FRACS:-1.00 0.50 0.25}"
N_EP="${N_EP:-1}"
SEED="${SEED:-1000}"

export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 TORCH_BLAS_PREFER_HIPBLASLT=0
export MOLMOACT2_DTYPE=bfloat16
LEROBOT_EVAL=lerobot-eval
[ -x /opt/libero-venv/bin/lerobot-eval ] && LEROBOT_EVAL=/opt/libero-venv/bin/lerobot-eval

echo "suite=$SUITE fracs=[$FRACS] n_ep/task=$N_EP  $(date '+%F %T')"
for f in $FRACS; do
  odir="$OUT/run_${SUITE}_${f}"
  log="$OUT/run_${SUITE}_${f}.log"
  mkdir -p "$odir"
  echo "===== groupdrop keep=$f  $(date '+%F %T') ====="
  VIS_GROUPDROP_KEEP_FRAC="$f" "$LEROBOT_EVAL" \
    --policy.type=molmoact2 \
    --policy.checkpoint_path="$CKPT" \
    --policy.inference_action_mode=continuous \
    --policy.enable_depth_reasoning=False \
    --policy.enable_adaptive_depth=False \
    --policy.enable_cuda_graph=False \
    --policy.num_steps=4 \
    --policy.norm_tag=libero \
    --policy.device=cuda \
    --env.type=libero \
    --env.task="$SUITE" \
    --eval.batch_size=1 \
    --eval.n_episodes="$N_EP" \
    --seed="$SEED" \
    --output_dir="$odir" > "$log" 2>&1
  rc=$?
  pc=$(grep -o "'pc_success': [0-9.]*" "$log" | tail -1)
  echo "[result] keep=$f  $pc  (rc=$rc)"
done
echo "DONE $(date '+%F %T')"
