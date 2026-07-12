#!/usr/bin/env bash
#SBATCH --job-name=droid_prep
#SBATCH -p mi210
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 01:30:00
#SBATCH --output=/shared_nobackup/%u/molmoact2/droid_prep_%j.log
# Download a bounded DROID subset and extract strided frames to JPEG (ONLINE) for
# the hybrid_droid distillation. Runs in the eval SIF on mi210 (has internet+PyAV+PIL).
set -u
U=$USER
WORK=/shared_nobackup/$U/molmoact2
SIF=$WORK/images/molmoact2-lerobot-eval.sif
CACHE=$WORK/hf_cache
CODE=$WORK/vit_distill
echo "[droid_prep] host=$(hostname) job=${SLURM_JOB_ID:-none} t=$(date -Is)"
apptainer exec --cleanenv --home "$CACHE":/cache \
  --env HF_HOME=/cache --env XDG_CACHE_HOME=/cache \
  --env HF_HUB_OFFLINE=0 --env TRANSFORMERS_OFFLINE=0 \
  --env DROID_OUT=/cache/droid_frames \
  --env DROID_SHARDS="${DROID_SHARDS:-3}" --env DROID_STRIDE="${DROID_STRIDE:-12}" \
  --env DROID_MAX_FRAMES="${DROID_MAX_FRAMES:-80000}" \
  --bind "$CODE":/work \
  "$SIF" bash -lc 'python /work/droid_prep.py' 2>&1 | tail -40
echo "[droid_prep] DONE rc=$? t=$(date -Is)"
