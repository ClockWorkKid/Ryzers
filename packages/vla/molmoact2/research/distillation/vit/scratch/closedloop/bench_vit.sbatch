#!/usr/bin/env bash
#SBATCH --job-name=vd_bench
#SBATCH -p mi210
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 00:30:00
#SBATCH --output=/shared_nobackup/%u/molmoact2/vd_bench_%x_%j.log
set -u
U=$USER
CACHE=/shared_nobackup/$U/molmoact2/hf_cache
OUT=/shared_nobackup/$U/molmoact2/outputs
SIF=/shared_nobackup/$U/molmoact2/images/molmoact2-lerobot-eval.sif
cp -f "$HOME/bench_vit_latency.py" "$OUT/bench_vit_latency.py"; sed -i 's/\r$//' "$OUT/bench_vit_latency.py"
echo "[vd_bench] host=$(hostname) job=${SLURM_JOB_ID:-none} rocr=${ROCR_VISIBLE_DEVICES:-} t=$(date -Is)"
[ -f "$SIF" ] || { echo "[vd_bench] SIF missing: $SIF"; exit 2; }
apptainer exec --rocm --cleanenv --home "$CACHE":/cache \
  --env USER="$U" --env LOGNAME="$U" \
  --env HF_HOME=/cache --env XDG_CACHE_HOME=/cache \
  --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
  --env TORCH_BLAS_PREFER_HIPBLASLT=0 \
  --env PYTORCH_ALLOC_CONF=expandable_segments:True \
  --bind "$OUT":/outputs \
  "$SIF" bash -lc "python /outputs/bench_vit_latency.py"
echo "[vd_bench] DONE rc=$? t=$(date -Is)"
