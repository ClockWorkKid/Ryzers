#!/usr/bin/env bash
#SBATCH --job-name=llm_qatk
#SBATCH -p gpu            # set to your GPU partition (runs were on MI210 nodes)
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 03:55:00
#SBATCH --output=llm_qatk_%x_%j.log
# ONE LLM-backbone QAT cell, CHUNKED to fit a ~4h scheduler wall-time cap. Runs
# BUDGET_MIN minutes, checkpoints latest.pt and exits(0); an afterany chain round
# then --resumes SEAMLESSLY (device-move only, NO re-calibration -- re-calibrating
# at each chunk boundary perturbs the trained scales and compounds into
# divergence). Idempotent: if the final ckpt already exists, the round is a no-op.
#   W=4 A=6 STEPS=6000 OUTTAG=mixedft AELR=5e-5 \
#     sbatch -p gpu --export=ALL llm_qat_cell_chunk.sbatch
#
# Paths below are overridable via env; point WORK at your own layout.
set -u
U=$USER
WORK="${WORK:-$HOME/molmoact2}"
SIF="${SIF:-$WORK/images/molmoact2-lerobot-eval.sif}"
CODE="${CODE:-$WORK/llm_distill}"
CACHE="${CACHE:-$WORK/hf_cache}"
PYLIBS="${PYLIBS:-$WORK/pylibs_quant}"
OV="${OV:-$HOME/roi_lerobot_overlay}"
MP=/opt/lerobot/src/lerobot/policies/molmoact2

W="${W:?set W}"; A="${A:?set A}"; IO="${IO:-8}"
STEPS="${STEPS:-6000}"
QDTYPE="${QDTYPE:-bf16}"
# mixed-precision scheme: quantize the projections, keep attention math in bf16.
QGROUPS="${QGROUPS:-attn_q,attn_kv,mlp}"
OUTTAG="${OUTTAG:-mixedft}"
BUDGET_MIN="${BUDGET_MIN:-200}"
TAG="W${W}A${A}"
OUTDIR_HOST="$WORK/outputs/llm_quant/${OUTTAG}_w${W}a${A}_io${IO}"
OUTDIR="/outputs/llm_quant/${OUTTAG}_w${W}a${A}_io${IO}"
log="$WORK/llm_qatk_${OUTTAG}_w${W}a${A}_io${IO}.log"
J="${SLURM_JOB_ID:-x}"

# done-guard: final ckpt present => nothing to do
if [ -f "$OUTDIR_HOST/llm_${TAG}_qat_ep.pt" ]; then
  echo "[llm_qatk] $TAG already COMPLETE ($OUTDIR_HOST/llm_${TAG}_qat_ep.pt); skip. t=$(date -Is)"
  exit 0
fi
# resume-guard: resume from latest.pt if a prior chunk left one
RESUME_ARGS=""
if [ -f "$OUTDIR_HOST/latest.pt" ]; then
  RESUME_ARGS="--resume $OUTDIR/latest.pt"
  echo "[llm_qatk] $TAG resuming from latest.pt"
fi

echo "[llm_qatk] host=$(hostname) $TAG steps=$STEPS budget=$BUDGET_MIN resume='${RESUME_ARGS}' job=$J t=$(date -Is)"
mkdir -p "$CACHE/miopen/job$J"
apptainer exec --rocm --cleanenv --home "$CACHE":/cache \
  --env USER="$U" --env LOGNAME="$U" --env HF_HOME=/cache --env XDG_CACHE_HOME=/cache \
  --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env HF_HUB_DOWNLOAD_TIMEOUT=120 \
  --env TORCH_BLAS_PREFER_HIPBLASLT=0 --env PYTORCH_ALLOC_CONF=expandable_segments:True \
  --env MIOPEN_USER_DB_PATH=/cache/miopen/job$J --env MIOPEN_CUSTOM_CACHE_DIR=/cache/miopen/job$J \
  --env HIP_VISIBLE_DEVICES=0 --env PYTHONPATH=/work:/pylibs \
  --bind "$CODE":/work --bind "$WORK/outputs":/outputs --bind "$PYLIBS":/pylibs --pwd /work \
  --bind "$OV/roi_prune.py":"$MP/roi_prune.py":ro \
  --bind "$OV/modeling_molmoact2.py":"$MP/modeling_molmoact2.py":ro \
  --bind "$OV/configuration_molmoact2.py":"$MP/configuration_molmoact2.py":ro \
  --bind "$OV/hf_modeling_molmoact2.py":"$MP/molmoact2_hf_model/modeling_molmoact2.py":ro \
  "$SIF" python /work/llm_qat.py \
    --weight-bits "$W" --act-bits "$A" --io-bits "$IO" --quant-dtype "$QDTYPE" \
    --groups "$QGROUPS" --target-steps "$STEPS" --out "$OUTDIR" \
    --batch "${BATCH:-2}" --grad-accum "${GACC:-4}" --ae-lr "${AELR:-5e-5}" \
    --lr-schedule "${LRSCHED:-cosine}" --val-frac "${VALFRAC:-0.05}" \
    --eval-every "${EVAL_EVERY:-500}" --save-every "${SAVE_EVERY:-250}" \
    --log-every "${LOG_EVERY:-25}" --calib-batches "${CALIB:-16}" \
    --heldout-batches "${HELDOUT:-32}" \
    --budget-min "$BUDGET_MIN" --num-workers "${WORKERS:-6}" \
    $RESUME_ARGS >> "$log" 2>&1
rc=$?
echo "[llm_qatk] $TAG rc=$rc t=$(date -Is)"
exit $rc
