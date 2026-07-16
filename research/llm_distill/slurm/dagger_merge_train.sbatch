#!/usr/bin/env bash
#SBATCH --job-name=dagger_train
#SBATCH -p gpu                       # <- set to your Slurm GPU partition (runs used a single MI210)
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -c 16
#SBATCH --mem=200G
#SBATCH -t 08:00:00
#SBATCH --output=logs/dagger_train_%x_%j.log
# Stage 3, steps 3 (merge + train): THE 47.5% RECIPE.
#   (1) merge the base ckpt (legacy/coadapt/lora) into a plain student+AE init (merge_base.py),
#   (2) fine-tune student LLM + LoRA-adapted action expert (--dagger-adapt-ae lora) with the
#       KV anchor ON, on a 50/50 on-policy/demo mix (--dagger-mix 0.5), 6k steps.
# Set ADAPT_AE=none for the frozen-AE ablation, or full for whole-AE (not recommended).
# Env: BASE, INIT, RELABEL, OUT, ADAPT_AE, STEPS, MIX.
set -u
U=$USER
WORK="${WORK:-$HOME/molmoact2}"
SIF="${SIF:-$WORK/images/molmoact2-lerobot-eval.sif}"
CODE="${CODE:-$WORK/llm_distill}"
CACHE="${CACHE:-$WORK/hf_cache}"
OUTPUTS="${OUTPUTS:-$WORK/outputs}"
OV="${OV:-$WORK/roi_overlay}"
MP=/opt/lerobot/src/lerobot/policies/molmoact2
GPU=$(echo "${ROCR_VISIBLE_DEVICES:-0}" | awk -F, '{print $1}')
BASE="${BASE:-/outputs/llm_distill/runs/qwen06w_expd/step_12000.pt}"
INIT="${INIT:-/outputs/llm_distill/dagger/expd/merged_init.pt}"
RELABEL="${RELABEL:-/outputs/llm_distill/dagger/expd/relabel}"
OUT="${OUT:-/outputs/llm_distill/dagger/expd/train_aelora}"
ADAPT_AE="${ADAPT_AE:-lora}"; STEPS="${STEPS:-6000}"; MIX="${MIX:-0.5}"
echo "[dagger_train] host=$(hostname) job=${SLURM_JOB_ID:-none} base=$BASE adapt_ae=$ADAPT_AE steps=$STEPS gpu=$GPU t=$(date -Is)"
apptainer exec --rocm --cleanenv --home "$CACHE":/cache \
  --env USER="$U" --env LOGNAME="$U" \
  --env HF_HOME=/cache --env XDG_CACHE_HOME=/cache --env MPLCONFIGDIR=/cache/mpl \
  --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
  --env HF_HUB_DOWNLOAD_TIMEOUT=120 --env TORCH_BLAS_PREFER_HIPBLASLT=0 \
  --env PYTORCH_ALLOC_CONF=expandable_segments:True \
  --env OMP_NUM_THREADS=1 --env MKL_NUM_THREADS=1 \
  --env HIP_VISIBLE_DEVICES=$GPU --env PYTHONPATH=/work \
  --bind "$CODE":/work --bind "$OUTPUTS":/outputs --pwd /work \
  --bind "$OV/roi_prune.py":"$MP/roi_prune.py":ro \
  --bind "$OV/modeling_molmoact2.py":"$MP/modeling_molmoact2.py":ro \
  --bind "$OV/configuration_molmoact2.py":"$MP/configuration_molmoact2.py":ro \
  --bind "$OV/hf_modeling_molmoact2.py":"$MP/molmoact2_hf_model/modeling_molmoact2.py":ro \
  "$SIF" bash -lc "set -e; \
    echo '=== merge base -> plain init ==='; \
    python /work/merge_base.py --base-ckpt '$BASE' --out '$INIT' --device cuda; \
    echo '=== dagger train (student LLM + AE-$ADAPT_AE, anchor on, 50/50 mix) ==='; \
    python /work/train_joint.py --preset qwen06w --init '$INIT' --out-dir '$OUT' \
      --train-student-only --dagger-dir '$RELABEL' --dagger-mix $MIX --dagger-adapt-ae $ADAPT_AE \
      --anchor-weight 1.0 --anchor-mode both --anchor-beta 0.1 --keep-teacher \
      --steps $STEPS --batch 4 --grad-accum 1 --lr 1e-4 \
      --log-every 20 --eval-every 200 --save-every 500 --num-workers 8"
RC=$?
echo "[dagger_train] DONE rc=$RC t=$(date -Is)"
exit $RC
