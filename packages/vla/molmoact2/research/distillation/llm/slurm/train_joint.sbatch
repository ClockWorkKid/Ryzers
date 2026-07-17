#!/usr/bin/env bash
#SBATCH --job-name=llmd_train
#SBATCH -p gpu                       # <- set to your Slurm GPU partition (runs used a single MI210)
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -c 32
#SBATCH --mem=200G
#SBATCH -t 08:00:00
#SBATCH --output=logs/llmd_train_%x_%j.log
# Joint distillation trainer driver (student LLM + action-expert LoRA, flow-matching loss,
# + optional all-layer teacher-KV anchor). This single driver expresses every training
# stage of the winning route via env vars / EXTRA_ARGS:
#
#   Stage 1 (KV-anchored functional distillation):
#     ANCHOR_W=1.0 KEEP_TEACHER=1 STEPS=6000 OUTDIR=/outputs/llm_distill/runs/qwen06w_anchor sbatch train_joint.sbatch
#   Stage 2 (LoRA fine-tune of {llm,ae,vit} on real LIBERO, anchor off):
#     EXTRA_ARGS="--lora-finetune --lora-targets llm,ae,vit" STEPS=12000 sbatch train_joint.sbatch
#   Stage 3 (on-policy DAgger, LoRA-adapted action expert -> the 47.5% run):  see dagger_merge_train.sbatch
#
# Overlay-bound teacher (train/deploy consistency). Set WORK + OV; see ../docker/README.md.
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
OUTDIR="${OUTDIR:-/outputs/llm_distill/runs/${PRESET}_fn}"
# KV-anchor knobs (Stage 1). ANCHOR_W=0 disables the anchor (plain functional).
ANCHOR_W="${ANCHOR_W:-0.0}"; ANCHOR_MODE="${ANCHOR_MODE:-both}"; ANCHOR_BETA="${ANCHOR_BETA:-0.1}"
KEEP=""; [ "${KEEP_TEACHER:-0}" = "1" ] && KEEP="--keep-teacher"
GPU=$(echo "${ROCR_VISIBLE_DEVICES:-0}" | awk -F, '{print $1}')
echo "[train] host=$(hostname) job=${SLURM_JOB_ID:-none} preset=$PRESET steps=${STEPS:-4000} batch=${BATCH:-8} anchor=$ANCHOR_W gpu=$GPU t=$(date -Is)"
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
  "$SIF" python /work/train_joint.py \
    --preset "$PRESET" --init "$INIT" --out-dir "$OUTDIR" \
    --steps "${STEPS:-4000}" --batch "${BATCH:-8}" --grad-accum "${GACC:-1}" \
    --lr "${LR:-2e-4}" --lora-lr "${LORA_LR:-2e-4}" \
    --anchor-weight "$ANCHOR_W" --anchor-mode "$ANCHOR_MODE" --anchor-beta "$ANCHOR_BETA" $KEEP \
    --eval-every "${EVAL_EVERY:-200}" --save-every "${SAVE_EVERY:-500}" \
    --log-every "${LOG_EVERY:-20}" --lora-rank "${LORA_RANK:-16}" \
    --num-workers "${NUM_WORKERS:-8}" ${EXTRA_ARGS:-}
RC=$?
echo "[train] DONE rc=$RC t=$(date -Is)"
exit $RC
