#!/usr/bin/env bash
#SBATCH --job-name=llmd_warm
#SBATCH -p gpu                       # <- set to your Slurm GPU partition (runs used a single MI210)
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -c 16
#SBATCH --mem=120G
#SBATCH -t 00:30:00
#SBATCH --output=logs/llmd_warm_%x_%j.log
# Stage 0: build the Qwen3-0.6B -> qwen06w student warm-start init, then re-run the
# Module-1 student validation with the warm weights loaded (finite KV + informational
# cosine-to-teacher). Everything runs inside the eval SIF with the model overlay bound.
#
# Paths are env-overridable so this is portable across clusters. Set WORK (workspace root
# holding images/, llm_distill/, hf_cache/, outputs/) and OV (model overlay dir). See
# ../docker/README.md for how to build the SIF and where the overlay files come from.
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
OUT="${OUT:-/outputs/llm_distill/warmstart/${PRESET}_init.pt}"
GPU=$(echo "${ROCR_VISIBLE_DEVICES:-0}" | awk -F, '{print $1}')
echo "[warm] host=$(hostname) job=${SLURM_JOB_ID:-none} preset=$PRESET out=$OUT gpu=$GPU t=$(date -Is)"
COMMON=(--rocm --cleanenv --home "$CACHE":/cache
  --env USER="$U" --env LOGNAME="$U"
  --env HF_HOME=/cache --env XDG_CACHE_HOME=/cache
  --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1
  --env HF_HUB_DOWNLOAD_TIMEOUT=120 --env TORCH_BLAS_PREFER_HIPBLASLT=0
  --env PYTORCH_ALLOC_CONF=expandable_segments:True
  --env HIP_VISIBLE_DEVICES=$GPU --env PYTHONPATH=/work
  --bind "$CODE":/work --bind "$OUTPUTS":/outputs --pwd /work
  --bind "$OV/roi_prune.py":"$MP/roi_prune.py":ro
  --bind "$OV/modeling_molmoact2.py":"$MP/modeling_molmoact2.py":ro
  --bind "$OV/configuration_molmoact2.py":"$MP/configuration_molmoact2.py":ro
  --bind "$OV/hf_modeling_molmoact2.py":"$MP/molmoact2_hf_model/modeling_molmoact2.py":ro)

echo "=== build warm-start ==="
apptainer exec "${COMMON[@]}" "$SIF" python /work/warmstart.py --preset "$PRESET" --out "$OUT"
RC=$?
[ $RC -eq 0 ] || { echo "[warm] warmstart FAILED rc=$RC"; exit $RC; }

echo "=== validate warm-started student ==="
apptainer exec "${COMMON[@]}" "$SIF" python /work/validate_student.py \
  --preset "$PRESET" --init "$OUT" --batch 2 --device cuda --dtype bf16
RC=$?
echo "[warm] DONE rc=$RC t=$(date -Is)"
exit $RC
