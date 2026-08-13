#!/usr/bin/env bash
#SBATCH --job-name=qatft210
#SBATCH -p gpu            # single-GPU partition (runs used MI210 nodes)
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 04:00:00
#SBATCH --output=qatft_%x_%j.log
# Finetune the *fake-quant W4A6* student ViT + LoRA(VLM) + action expert on LIBERO
# via apptainer + eval SIF on a single GPU. Brevitas + quant/distill pkgs are
# BIND-MOUNTED (/pylibs +
# /pkg) -- no image rebuild. One 3h50m graceful-capped segment; resumes from
# `last`, so an afterany chain finishes the full 12k-step run.
#   Validation :  sbatch --export=ALL,VARIANT=cnn,STEPS=30,SAVE_FREQ=10,SEG_SECONDS=1500 qat_ft_mi210.sbatch
#   Full (cnn) :  sbatch --export=ALL,VARIANT=cnn qat_ft_mi210.sbatch
#   Full (tv)  :  sbatch --export=ALL,VARIANT=tinyvit qat_ft_mi210.sbatch
set -u
U=$USER
WORK="${WORK:-$HOME/molmoact2}"
CACHE=$WORK/hf_cache
OUT=$WORK/outputs
SIF=$WORK/images/molmoact2-lerobot-eval.sif
OV=$HOME/roi_lerobot_overlay
MP=/opt/lerobot/src/lerobot/policies/molmoact2
PYLIBS=$WORK/pylibs_quant
PKG=$OUT/quantpkg

VARIANT="${VARIANT:-cnn}"
case "$VARIANT" in
  cnn)     FP32=/outputs/cnn_fpga_full.pt;    QAT="${QAT_STATE:-/outputs/quant/cnn_w4a6_qat.pt}" ;;
  tinyvit) FP32=/outputs/siglip_nano_full.pt; QAT="${QAT_STATE:-/outputs/quant/tinyvit_w4a6_qatfix_ds.pt}" ;;
  *) echo "[qatft210] unknown VARIANT=$VARIANT (want cnn|tinyvit)"; exit 3 ;;
esac

RUN="${RUN:-qat_ft_$VARIANT}"
STEPS="${STEPS:-12000}"
SAVE_FREQ="${SAVE_FREQ:-2000}"          # MUST be < steps-per-segment else the chain restarts from 0
BS="${BS:-8}"
UNFREEZE="${UNFREEZE:-4}"
WEIGHT_BITS="${WEIGHT_BITS:-4}"
ACT_BITS="${ACT_BITS:-6}"
QAT_STUDENT_TRAINABLE="${QAT_STUDENT_TRAINABLE:-1}"
PORT="${PORT:-29782}"
CAP="${SEG_SECONDS:-13800}"             # ~3h50m graceful stop before the 4h wall

GPU=$(echo "${ROCR_VISIBLE_DEVICES:-0}" | awk -F, '{print $1}')
RUNDIR=$OUT/$RUN
LAST=$RUNDIR/checkpoints/last/pretrained_model
LOG=$OUT/${RUN}.$(date +%s).log
DONE=$(printf '%06d' "$STEPS")

# promote the launcher into the bind target from HOME if present
[ -f "$HOME/run_train_qat_student.py" ] && { cp -f "$HOME/run_train_qat_student.py" "$OUT/"; sed -i 's/\r$//' "$OUT/run_train_qat_student.py"; }

echo "[qatft210] host=$(hostname) job=${SLURM_JOB_ID:-none} run=$RUN variant=$VARIANT W${WEIGHT_BITS}A${ACT_BITS} trainable=$QAT_STUDENT_TRAINABLE gpu=$GPU t=$(date -Is)"
echo "[qatft210] fp32=$FP32 qat=$QAT steps=$STEPS bs=$BS save_freq=$SAVE_FREQ cap=${CAP}s"
[ -f "$SIF" ] || { echo "[qatft210] SIF missing: $SIF"; exit 2; }
[ -f "$OUT/run_train_qat_student.py" ] || { echo "[qatft210] missing /outputs/run_train_qat_student.py"; exit 4; }
[ -f "$PKG/quant/quant_student.py" ] || { echo "[qatft210] missing $PKG/quant/quant_student.py"; exit 4; }
[ -d "$PYLIBS/brevitas" ] || { echo "[qatft210] missing brevitas at $PYLIBS"; exit 4; }
if [ -d "$RUNDIR/checkpoints/$DONE" ]; then echo "[qatft210] $RUN already at $STEPS -> skip"; exit 0; fi

COMMON="--policy.type=molmoact2 --policy.checkpoint_path=allenai/MolmoAct2-LIBERO --policy.push_to_hub=false \
  --policy.enable_lora_vlm=true --policy.action_mode=both \
  --policy.chunk_size=10 --policy.n_action_steps=10 --policy.model_dtype=bfloat16 \
  --policy.num_flow_timesteps=8 --policy.gradient_checkpointing=true \
  --policy.roi_prune_enable=false --policy.roi_fastv_enable=false \
  --policy.roi_unfreeze_last_decoder=$UNFREEZE \
  --dataset.repo_id=allenai/MolmoAct2-LIBERO-Dataset --dataset.revision=main --dataset.video_backend=pyav \
  --batch_size=$BS --steps=$STEPS --save_freq=$SAVE_FREQ --log_freq=25"

if [ -f "$LAST/train_config.json" ]; then
  echo "[qatft210] RESUME $RUN"
  TRAIN="/outputs/run_train_qat_student.py --config_path=/outputs/$RUN/checkpoints/last/pretrained_model/train_config.json --resume=true"
else
  echo "[qatft210] FRESH  $RUN"
  rm -rf "$RUNDIR" 2>/dev/null || true
  TRAIN="/outputs/run_train_qat_student.py $COMMON --output_dir=/outputs/$RUN"
fi

timeout --signal=TERM "$CAP" apptainer exec --rocm --cleanenv --home "$CACHE":/cache \
  --env USER="$U" --env LOGNAME="$U" \
  --env HF_HOME=/cache --env XDG_CACHE_HOME=/cache \
  --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
  --env HF_HUB_DOWNLOAD_TIMEOUT=120 --env TORCH_BLAS_PREFER_HIPBLASLT=0 \
  --env PYTORCH_ALLOC_CONF=expandable_segments:True \
  --env HIP_VISIBLE_DEVICES=$GPU \
  --env PYTHONPATH=/pylibs:/pkg \
  --env QUANT_VARIANT="$VARIANT" --env FP32_STUDENT="$FP32" --env QAT_STATE="$QAT" \
  --env WEIGHT_BITS="$WEIGHT_BITS" --env ACT_BITS="$ACT_BITS" \
  --env QAT_STUDENT_TRAINABLE="$QAT_STUDENT_TRAINABLE" \
  --bind "$OUT":/outputs --bind "$PYLIBS":/pylibs --bind "$PKG":/pkg \
  --bind "$OV/roi_prune.py":"$MP/roi_prune.py":ro \
  --bind "$OV/modeling_molmoact2.py":"$MP/modeling_molmoact2.py":ro \
  --bind "$OV/configuration_molmoact2.py":"$MP/configuration_molmoact2.py":ro \
  --bind "$OV/hf_modeling_molmoact2.py":"$MP/molmoact2_hf_model/modeling_molmoact2.py":ro \
  "$SIF" bash -lc "accelerate launch --num_processes=1 --mixed_precision=bf16 --main_process_port=$PORT $TRAIN" > "$LOG" 2>&1
RC=$?
echo "[qatft210] SEGMENT_DONE $RUN rc=$RC (124=cap-reached, resume next segment) log=$LOG t=$(date -Is)"
[ "$RC" = "124" ] && exit 0
exit "$RC"
