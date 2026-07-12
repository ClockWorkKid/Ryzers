#!/usr/bin/env bash
#SBATCH --job-name=llmd_s1
#SBATCH -p mi325x
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -c 12
#SBATCH --mem=96G
#SBATCH -t 02:00:00
#SBATCH --output=/shared_nobackup/%u/molmoact2/llmd_stage1_%x_%j.log
# Run D Stage-1: LLM-only curriculum distillation (thin-twin student <- frozen teacher
# per-layer hidden + KV + final) on real LIBERO. mi325x = fast gfx942 (2h QOS cap) -> ~10k
# steps in one segment at ~1.7 it/s. Teacher targets computed on the fly (frozen, bf16).
set -u
U=$USER
WORK=/shared_nobackup/$U/molmoact2
SIF=$WORK/images/molmoact2-lerobot-eval.sif
CODE=$WORK/llm_distill
CACHE=$WORK/hf_cache
STEPS="${STEPS:-10000}"
BATCH="${BATCH:-4}"
SAVE="${SAVE:-2000}"
OUT="${OUT:-/outputs/llm_distill/stage1}"
GPU=$(echo "${ROCR_VISIBLE_DEVICES:-0}" | awk -F, '{print $1}')
ulimit -n 65536 2>/dev/null || true
echo "[llmd_s1] host=$(hostname) job=${SLURM_JOB_ID:-none} steps=$STEPS batch=$BATCH gpu=$GPU nofile=$(ulimit -n) t=$(date -Is)"
mkdir -p "$WORK/outputs/llm_distill/stage1"

apptainer exec --rocm --cleanenv --home "$CACHE":/cache \
  --env USER="$U" --env LOGNAME="$U" \
  --env HF_HOME=/cache --env XDG_CACHE_HOME=/cache \
  --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
  --env HF_HUB_DOWNLOAD_TIMEOUT=120 --env TORCH_BLAS_PREFER_HIPBLASLT=0 \
  --env PYTORCH_ALLOC_CONF=expandable_segments:True \
  --env HIP_VISIBLE_DEVICES=$GPU \
  --env PYTHONPATH=/work --bind "$CODE":/work --bind "$WORK/outputs":/outputs --pwd /work \
  "$SIF" python train_stage1.py \
    --steps $STEPS --batch $BATCH --lr 3e-4 --warmup 300 \
    --dtype bf16 --device cuda --save-every $SAVE --log-every 25 \
    --w-hidden 1.0 --w-kv 2.0 --w-final 1.0 --out "$OUT"
RC=$?
echo "[llmd_s1] DONE rc=$RC t=$(date -Is)"
exit $RC
