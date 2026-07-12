#!/usr/bin/env bash
#SBATCH --job-name=vd_tsmoke
#SBATCH -p mi210
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -c 32
#SBATCH -t 00:30:00
#SBATCH --output=/shared_nobackup/%u/molmoact2/vd_tsmoke_%j.log
# Pristine-teacher eval smoke on MI210 (same SKU used for the full comparison).
set -u
U=$USER
CACHE=/shared_nobackup/$U/molmoact2/hf_cache
OUT=/shared_nobackup/$U/molmoact2/outputs
SIF=/shared_nobackup/$U/molmoact2/images/molmoact2-lerobot-eval.sif
mkdir -p "$OUT/vd_smoke"
cp -f "$HOME/cl_teacher_smoke_inner.sh" "$OUT/cl_teacher_smoke_inner.sh"
sed -i 's/\r$//' "$OUT/cl_teacher_smoke_inner.sh"
echo "[tsmoke] host=$(hostname) job=${SLURM_JOB_ID:-none} rocr=${ROCR_VISIBLE_DEVICES:-} t=$(date -Is)"
[ -f "$SIF" ] || { echo "[tsmoke] SIF missing: $SIF"; exit 2; }
apptainer exec --rocm --cleanenv --home "$CACHE":/cache \
  --env USER="$U" --env LOGNAME="$U" \
  --env HF_HOME=/cache --env XDG_CACHE_HOME=/cache \
  --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
  --env HF_HUB_DOWNLOAD_TIMEOUT=120 --env TORCH_BLAS_PREFER_HIPBLASLT=0 \
  --env PYTORCH_ALLOC_CONF=expandable_segments:True \
  --env MUJOCO_GL=osmesa --env PYOPENGL_PLATFORM=osmesa \
  --bind "$OUT":/outputs \
  "$SIF" bash -lc "bash /outputs/cl_teacher_smoke_inner.sh"
echo "[tsmoke] DONE rc=$? t=$(date -Is)"
