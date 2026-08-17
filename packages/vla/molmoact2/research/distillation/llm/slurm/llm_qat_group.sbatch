#!/usr/bin/env bash
#SBATCH --job-name=llm_qatg
#SBATCH -p gpu            # set to your 4-GPU partition (runs were on MI210 nodes)
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH -c 64
#SBATCH --mem=256G
#SBATCH -t 03:55:00
#SBATCH --output=llm_qatg_%x_%j.log
# Run a GROUP of up to 4 LLM-backbone PTQ/QAT arms on one 4-GPU node (one W/A
# cell per GPU). STEPS=0 => PTQ-only (fast); STEPS>0 => QAT fine-tuning. Chunked
# to BUDGET_MIN (saves latest.pt for --dependency=afterany resume).
#   ARMS="8:8:8 6:6:8 4:6:8 4:4:8" STEPS=0 OUTTAG=uniform \
#     QGROUPS=attn_q,attn_kv,attn_sdpa,mlp sbatch --export=ALL llm_qat_group.sbatch
# arm spec = weightbits:actbits:iobits  (iobits=8 defensive, -1 = quantize I/O to target)
#
# Paths below are overridable via env; defaults assume a per-user work tree with
# an Apptainer SIF, an HF cache, the quant python libs, and the MolmoAct2 policy
# overlay. Point WORK at your own layout.
set -u
U=$USER
WORK="${WORK:-$HOME/molmoact2}"
SIF="${SIF:-$WORK/images/molmoact2-lerobot-eval.sif}"
CODE="${CODE:-$WORK/llm_distill}"
CACHE="${CACHE:-$WORK/hf_cache}"
PYLIBS="${PYLIBS:-$WORK/pylibs_quant}"
OV="${OV:-$HOME/roi_lerobot_overlay}"
MP=/opt/lerobot/src/lerobot/policies/molmoact2
ARMS="${ARMS:?set ARMS as w:a:io e.g. '8:8:8 6:6:8 4:6:8 4:4:8'}"
STEPS="${STEPS:-0}"
BUDGET_MIN="${BUDGET_MIN:-210}"
QDTYPE="${QDTYPE:-bf16}"
RESUME="${RESUME:-0}"
# uniform scheme = attn_q,attn_kv,attn_sdpa,mlp; the mixed-precision scheme drops
# attn_sdpa (keeps the attention math in bf16), which the sensitivity ablation
# identifies as the fragile component.
QGROUPS="${QGROUPS:-attn_q,attn_kv,attn_sdpa,mlp}"
# OUTTAG namespaces the run dir (uniform vs mixed) so blobs don't clobber.
OUTTAG="${OUTTAG:-uniform}"
echo "[llm_qatg] host=$(hostname) arms='$ARMS' steps=$STEPS groups=$QGROUPS tag=$OUTTAG budget=$BUDGET_MIN resume=$RESUME job=${SLURM_JOB_ID:-none} t=$(date -Is)"
i=0; pids=()
for spec in $ARMS; do
  w="${spec%%:*}"; rest="${spec#*:}"; a="${rest%%:*}"; io="${rest#*:}"
  gpu=$((i % 4))
  OUTDIR="/outputs/llm_quant/${OUTTAG}_w${w}a${a}_io${io}"
  log="$WORK/llm_qatg_${OUTTAG}_w${w}a${a}_io${io}.log"
  RESUME_ARGS=""; [ "$RESUME" = "1" ] && RESUME_ARGS="--resume $OUTDIR/latest.pt"
  echo "[llm_qatg] launch arm=W${w}A${a} io=$io gpu=$gpu out=$OUTDIR log=$log"
  mkdir -p "$CACHE/miopen/gpu$gpu"
  (
    sleep $((i * 30))
    apptainer exec --rocm --cleanenv --home "$CACHE":/cache \
      --env USER="$U" --env LOGNAME="$U" --env HF_HOME=/cache --env XDG_CACHE_HOME=/cache \
      --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env HF_HUB_DOWNLOAD_TIMEOUT=120 \
      --env TORCH_BLAS_PREFER_HIPBLASLT=0 --env PYTORCH_ALLOC_CONF=expandable_segments:True \
      --env MIOPEN_USER_DB_PATH=/cache/miopen/gpu$gpu --env MIOPEN_CUSTOM_CACHE_DIR=/cache/miopen/gpu$gpu \
      --env HIP_VISIBLE_DEVICES=$gpu --env PYTHONPATH=/work:/pylibs \
      --bind "$CODE":/work --bind "$WORK/outputs":/outputs --bind "$PYLIBS":/pylibs --pwd /work \
      --bind "$OV/roi_prune.py":"$MP/roi_prune.py":ro \
      --bind "$OV/modeling_molmoact2.py":"$MP/modeling_molmoact2.py":ro \
      --bind "$OV/configuration_molmoact2.py":"$MP/configuration_molmoact2.py":ro \
      --bind "$OV/hf_modeling_molmoact2.py":"$MP/molmoact2_hf_model/modeling_molmoact2.py":ro \
      "$SIF" python /work/llm_qat.py \
        --weight-bits "$w" --act-bits "$a" --io-bits "$io" --quant-dtype "$QDTYPE" \
        --groups "$QGROUPS" --target-steps "$STEPS" --out "$OUTDIR" \
        --batch "${BATCH:-2}" --grad-accum "${GACC:-4}" --ae-lr "${AELR:-5e-5}" \
        --lr-schedule "${LRSCHED:-constant}" \
        --eval-every "${EVAL_EVERY:-500}" --save-every "${SAVE_EVERY:-500}" \
        --log-every "${LOG_EVERY:-25}" --calib-batches "${CALIB:-16}" \
        --budget-min "$BUDGET_MIN" --num-workers "${WORKERS:-6}" \
        $RESUME_ARGS > "$log" 2>&1
    rc=$?
    echo "[llm_qatg] arm=W${w}A${a} io=$io rc=$rc t=$(date -Is)"
    exit $rc
  ) &
  pids+=($!)
  i=$((i + 1))
done
# Propagate the worst arm exit code so a crashed arm fails the job (an afterok
# chain then HALTS instead of cascading through every resume round in minutes).
rc_max=0
for p in "${pids[@]}"; do wait "$p" || rc_max=$?; done
echo "[llm_qatg] ALL DONE arms='$ARMS' rc_max=$rc_max t=$(date -Is)"
ls -la "$WORK"/outputs/llm_quant/${OUTTAG}_w*_io*/*.pt 2>/dev/null
exit $rc_max
