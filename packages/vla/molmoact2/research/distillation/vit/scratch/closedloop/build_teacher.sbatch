#!/usr/bin/env bash
#SBATCH --job-name=vd_buildT
#SBATCH -p mi210
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -c 32
#SBATCH -t 00:40:00
#SBATCH --output=/shared_nobackup/%u/molmoact2/vd_buildT_%j.log
# Build the pristine teacher checkpoint (policy + processors w/ dataset stats).
set -u
U=$USER
CACHE=/shared_nobackup/$U/molmoact2/hf_cache
OUT=/shared_nobackup/$U/molmoact2/outputs
SIF=/shared_nobackup/$U/molmoact2/images/molmoact2-lerobot-eval.sif
cp -f "$HOME/build_teacher_ckpt.py" "$OUT/build_teacher_ckpt.py"
sed -i 's/\r$//' "$OUT/build_teacher_ckpt.py"
echo "[buildT] host=$(hostname) job=${SLURM_JOB_ID:-none} rocr=${ROCR_VISIBLE_DEVICES:-} t=$(date -Is)"
[ -f "$SIF" ] || { echo "[buildT] SIF missing: $SIF"; exit 2; }
apptainer exec --rocm --cleanenv --home "$CACHE":/cache \
  --env USER="$U" --env LOGNAME="$U" \
  --env HF_HOME=/cache --env XDG_CACHE_HOME=/cache \
  --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
  --env HF_HUB_DOWNLOAD_TIMEOUT=120 --env TORCH_BLAS_PREFER_HIPBLASLT=0 \
  --env PYTORCH_ALLOC_CONF=expandable_segments:True \
  --bind "$OUT":/outputs \
  "$SIF" bash -lc "python /outputs/build_teacher_ckpt.py"
echo "[buildT] DONE rc=$? t=$(date -Is)"
