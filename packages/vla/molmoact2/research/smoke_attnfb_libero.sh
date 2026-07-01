#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Closed-loop smoke for attention-feedback deterministic group-drop on
# libero_object: FULL vs RANDOM group-drop vs ATTNFB (ours) at keep=0.5.
# Resets + applies the patch stack (model + lerobot wrapper), then runs
# lerobot-eval per condition. Runs INSIDE the molmoact2 container.
set -uo pipefail
HF=/root/.cache/huggingface
P=/scripts
OUT=/ryzers/outputs/attnfb_smoke
mkdir -p "$OUT"
CKPT="${CKPT:-allenai/MolmoAct2-Think-LIBERO}"
SUITE="${SUITE:-libero_object}"
N_EP="${N_EP:-1}"
SEED="${SEED:-1000}"
KEEP="${KEEP:-0.5}"
RESET="${RESET:-1}"
CONDS="${CONDS:-full random attnfb}"
SURVEY_EVERY="${SURVEY_EVERY:-0}"

export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 TORCH_BLAS_PREFER_HIPBLASLT=0
export MOLMOACT2_DTYPE=bfloat16
LEROBOT_EVAL=lerobot-eval
[ -x /opt/libero-venv/bin/lerobot-eval ] && LEROBOT_EVAL=/opt/libero-venv/bin/lerobot-eval

if [ "$RESET" = "1" ]; then
  echo "===== reset modeling to pristine ====="
  python - <<'PY'
from huggingface_hub import hf_hub_download
for repo in ["allenai/MolmoAct2-Think-LIBERO", "allenai/MolmoAct2-DROID"]:
    try:
        print("redownloaded", hf_hub_download(repo, "modeling_molmoact2.py", force_download=True))
    except Exception as e:
        print("skip", repo, repr(e))
PY
  find "$HF/modules/transformers_modules" -name modeling_molmoact2.py -delete 2>/dev/null || true
fi

echo "===== apply patch stack ====="
python "$P/patch_vis_reduce.py" "$HF"
python "$P/patch_vis_groupdrop.py" "$HF"
python "$P/patch_vis_attnfeedback.py" "$HF"
# lerobot wrapper carrier (libero venv site-packages)
/opt/libero-venv/bin/python "$P/patch_lerobot_attnfb.py" || python "$P/patch_lerobot_attnfb.py"

run_cond () {
  local name="$1" keepfrac="$2" attnfb="$3" survey="${4:-0}"
  local odir="$OUT/run_${SUITE}_${name}" log="$OUT/run_${SUITE}_${name}.log"
  mkdir -p "$odir"
  echo "===== [$name] keep=$keepfrac attnfb=$attnfb survey_every=$survey  $(date '+%F %T') ====="
  env VIS_GROUPDROP_KEEP_FRAC="$keepfrac" VIS_ATTNFB="$attnfb" VIS_ATTNFB_SURVEY_EVERY="$survey" \
    "$LEROBOT_EVAL" \
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
  local rc=$? pc
  pc=$(grep -o "'pc_success': [0-9.]*" "$log" | tail -1)
  echo "[result] $name  $pc  (rc=$rc)"
}

# full baseline (no prune), random group-drop, attnfb deterministic
for c in $CONDS; do
  case "$c" in
    full)   run_cond full   1.00     "" 0 ;;
    random) run_cond random "$KEEP"  "" 0 ;;
    attnfb) for se in $SURVEY_EVERY; do run_cond "attnfb_se${se}" "$KEEP" 1 "$se"; done ;;
    *) echo "[warn] unknown cond '$c'";;
  esac
done

echo "DONE $(date '+%F %T')"
