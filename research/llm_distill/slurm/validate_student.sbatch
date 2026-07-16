#!/usr/bin/env bash
#SBATCH --job-name=llmd_valm1
#SBATCH -p gpu                       # <- set to your Slurm GPU partition (runs used a single MI210)
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -c 16
#SBATCH --mem=120G
#SBATCH -t 00:30:00
#SBATCH --output=logs/llmd_valm1_%x_%j.log
# Module-1 validation: StudentTextModel KV contract vs teacher on a REAL LIBERO batch.
# Binds the ROI overlay so the teacher modeling code == the eval/deploy path (train/deploy
# consistency). Runs inside the eval SIF. Set WORK + OV; see ../docker/README.md.
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
echo "[valm1] host=$(hostname) job=${SLURM_JOB_ID:-none} preset=${PRESET:-qwen06w} gpu=$GPU t=$(date -Is)"
for f in student.py data.py validate_student.py; do
  [ -f "$CODE/$f" ] || { echo "[valm1] missing $CODE/$f"; exit 3; }
done
apptainer exec --rocm --cleanenv --home "$CACHE":/cache \
  --env USER="$U" --env LOGNAME="$U" \
  --env HF_HOME=/cache --env XDG_CACHE_HOME=/cache \
  --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
  --env HF_HUB_DOWNLOAD_TIMEOUT=120 --env TORCH_BLAS_PREFER_HIPBLASLT=0 \
  --env PYTORCH_ALLOC_CONF=expandable_segments:True \
  --env HIP_VISIBLE_DEVICES=$GPU --env PYTHONPATH=/work \
  --bind "$CODE":/work --bind "$OUTPUTS":/outputs --pwd /work \
  --bind "$OV/roi_prune.py":"$MP/roi_prune.py":ro \
  --bind "$OV/modeling_molmoact2.py":"$MP/modeling_molmoact2.py":ro \
  --bind "$OV/configuration_molmoact2.py":"$MP/configuration_molmoact2.py":ro \
  --bind "$OV/hf_modeling_molmoact2.py":"$MP/molmoact2_hf_model/modeling_molmoact2.py":ro \
  "$SIF" python /work/validate_student.py --preset "${PRESET:-qwen06w}" --batch 2 --device cuda --dtype bf16
RC=$?
echo "[valm1] DONE rc=$RC t=$(date -Is)"
exit $RC
