#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Stage-D PRE-ENCODER vision-token pruning ablation for MolmoAct2-Think-LIBERO.
# Sweeps VIS_PREENC_KEEP_FRAC across the 5 LIBERO suites in the bf16 fast path
# (depth off, num_steps=4), synchronous closed loop via lerobot-eval. 1 episode
# per task (per-task breakdown), libero_90 subsampled to a spread of tasks.
# Runs INSIDE the molmoact2 container (see run_preenc_detached.sh). Resumable:
# (suite,frac) combos whose eval_info.json exists are skipped.
set -uo pipefail

OUT=/ryzers/outputs/ablation_preenc
mkdir -p "$OUT"
RESULTS="$OUT/results.csv"
CKPT="${CKPT:-allenai/MolmoAct2-Think-LIBERO}"
NUM_STEPS="${NUM_STEPS:-4}"
N_EP="${N_EP:-1}"                 # episodes PER TASK
SEED="${SEED:-1000}"
SUITES="${SUITES:-libero_spatial libero_object libero_goal libero_10 libero_90}"
FRACS="${FRACS:-1.00 0.75 0.50 0.25}"
# libero_90 has 90 tasks; subsample a spread to keep runtime bounded.
L90_TASK_IDS="${L90_TASK_IDS:-[0,6,12,18,24,30,36,42,48,54,60,66,72,78,84]}"

export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 TORCH_BLAS_PREFER_HIPBLASLT=0
export MOLMOACT2_DTYPE=bfloat16

LEROBOT_EVAL=lerobot-eval
[ -x /opt/libero-venv/bin/lerobot-eval ] && LEROBOT_EVAL=/opt/libero-venv/bin/lerobot-eval

[ -f "$RESULTS" ] || echo "suite,keep_frac,pc_success,n_episodes,eval_s,eval_ep_s,status,timestamp" > "$RESULTS"

echo "########## PREENC ABLATION START $(date '+%F %T') ##########"
echo "suites=[$SUITES] fracs=[$FRACS] n_ep/task=$N_EP num_steps=$NUM_STEPS"

for suite in $SUITES; do
  for f in $FRACS; do
    name="run_${suite}_${f}"
    odir="$OUT/$name"
    log="$OUT/${name}.log"
    if [ -f "$odir/eval_info.json" ]; then
      echo "[skip] $suite frac=$f (eval_info.json exists)"; continue
    fi
    mkdir -p "$odir"
    echo "===== eval suite=$suite frac=$f n_ep/task=$N_EP $(date '+%F %T') ====="
    EXTRA=()
    if [ "$suite" = "libero_90" ]; then EXTRA+=(--env.task_ids="$L90_TASK_IDS"); fi
    VIS_PREENC_KEEP_FRAC="$f" "$LEROBOT_EVAL" \
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
      --env.task="$suite" \
      "${EXTRA[@]}" \
      --eval.batch_size=1 \
      --eval.n_episodes="$N_EP" \
      --seed="$SEED" \
      --output_dir="$odir" > "$log" 2>&1
    rc=$?
    python - "$log" "$suite" "$f" "$RESULTS" "$rc" <<'PY' || true
import re, sys, time
log, suite, frac, res, rc = sys.argv[1:6]
txt = open(log, errors="ignore").read()
pcs = re.findall(r"'pc_success':\s*([\d.]+)", txt)
nep = re.findall(r"'n_episodes':\s*(\d+)", txt)
es  = re.findall(r"'eval_s':\s*([\d.]+)", txt)
eps = re.findall(r"'eval_ep_s':\s*([\d.]+)", txt)
pc  = pcs[-1] if pcs else ""
ne  = nep[-1] if nep else ""
e_s = es[-1]  if es  else ""
e_p = eps[-1] if eps else ""
status = "done" if (pc != "" and rc == "0") else "failed"
with open(res, "a") as fh:
    fh.write(f"{suite},{frac},{pc},{ne},{e_s},{e_p},{status},{time.strftime('%F %T')}\n")
print(f"[eval] {suite} frac={frac} pc_success={pc} n={ne} status={status}")
PY
  done
done

echo "########## PREENC ABLATION DONE $(date '+%F %T') ##########"
touch "$OUT/_PREENC_DONE"
