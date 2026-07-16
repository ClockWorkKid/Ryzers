#!/usr/bin/env bash
#SBATCH --job-name=dagger_val
#SBATCH -p gpu                       # <- set to your Slurm GPU partition (runs used a single MI210)
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -c 16
#SBATCH --mem=120G
#SBATCH -t 00:30:00
#SBATCH --output=logs/dagger_val_%x_%j.log
# Fast relabel sanity: build the teacher and run the rigorous (pinned, averaged) relabel
# validation (teacher label vs demo action MSE, and self-consistency under the flow field).
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
echo "[dagger_val] host=$(hostname) job=${SLURM_JOB_ID:-none} gpu=$GPU t=$(date -Is)"
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
  "$SIF" python /work/relabel_teacher.py --validate --device cuda
RC=$?
echo "[dagger_val] DONE rc=$RC t=$(date -Is)"
exit $RC
