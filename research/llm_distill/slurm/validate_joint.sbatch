#!/usr/bin/env bash
#SBATCH --job-name=llmd_valj
#SBATCH -p gpu                       # <- set to your Slurm GPU partition (runs used a single MI210)
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -c 16
#SBATCH --mem=120G
#SBATCH -t 00:30:00
#SBATCH --output=logs/llmd_valj_%x_%j.log
# Module-4a: functional training path end-to-end (teacher flow baseline -> attach student ->
# student flow loss + backward + grad-flow checks). Overlay-bound teacher (deploy consistency).
set -u
U=$USER
WORK="${WORK:-$HOME/molmoact2}"
SIF="${SIF:-$WORK/images/molmoact2-lerobot-eval.sif}"
CODE="${CODE:-$WORK/llm_distill}"
CACHE="${CACHE:-$WORK/hf_cache}"
OUTPUTS="${OUTPUTS:-$WORK/outputs}"
OV="${OV:-$WORK/roi_overlay}"
MP=/opt/lerobot/src/lerobot/policies/molmoact2
PRESET="${PRESET:-qwen06w}"
INIT="${INIT:-/outputs/llm_distill/warmstart/${PRESET}_init.pt}"
GPU=$(echo "${ROCR_VISIBLE_DEVICES:-0}" | awk -F, '{print $1}')
echo "[valj] host=$(hostname) job=${SLURM_JOB_ID:-none} preset=$PRESET gpu=$GPU t=$(date -Is)"
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
  "$SIF" python /work/validate_joint.py \
    --preset "$PRESET" --init "$INIT" --batch "${BATCH:-2}" --device cuda --lora-rank "${LORA_RANK:-16}"
RC=$?
echo "[valj] DONE rc=$RC t=$(date -Is)"
exit $RC
