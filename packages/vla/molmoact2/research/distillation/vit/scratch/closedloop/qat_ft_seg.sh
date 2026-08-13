#!/usr/bin/env bash
# One finetune segment of the *fake-quant W4A6* student ViT + LoRA(VLM) + action
# expert on LIBERO -- the proven "student_ft" recipe applied to the quantized
# encoder. Uses docker + the training image (the MI300X training path). Brevitas + the
# quant/distill packages are BIND-MOUNTED (/pylibs + /pkg) -- no image rebuild.
# Segmented + resumable (graceful stop before the SLURM wall, resume next seg).
#
# Usage:  VARIANT=cnn      bash qat_ft_seg.sh
#         VARIANT=tinyvit  QAT_STATE=/outputs/quant/tinyvit_qatfix_ds.pt bash qat_ft_seg.sh
set -u
U=$USER
WORK="${WORK:-$HOME/molmoact2}"
CACHE=$WORK/hf_cache
OUT=$WORK/outputs
OV=$HOME/roi_lerobot_overlay
MP=/opt/lerobot/src/lerobot/policies/molmoact2
IMG="${IMG:-molmoact2-lerobot-train:rocm942}"
PYLIBS="${PYLIBS:-$WORK/pylibs_quant}"
PKG="${PKG:-$OUT/quantpkg}"

VARIANT="${VARIANT:-cnn}"
case "$VARIANT" in
  cnn)     FP32="${FP32:-/outputs/cnn_fpga_full.pt}";     QAT="${QAT_STATE:-/outputs/quant/cnn_w4a6_qat.pt}" ;;
  tinyvit) FP32="${FP32:-/outputs/siglip_nano_full.pt}";  QAT="${QAT_STATE:-/outputs/quant/tinyvit_w4a6_qatfix_ds.pt}" ;;
  *) echo "[qat-ft] unknown VARIANT=$VARIANT (want cnn|tinyvit)"; exit 2 ;;
esac

RUN="${RUN:-qat_ft_${VARIANT}}"
STEPS="${STEPS:-12000}"
SAVE_FREQ="${SAVE_FREQ:-1000}"
BS="${BS:-8}"
UNFREEZE="${UNFREEZE:-4}"
WEIGHT_BITS="${WEIGHT_BITS:-4}"
ACT_BITS="${ACT_BITS:-6}"
QAT_STUDENT_TRAINABLE="${QAT_STUDENT_TRAINABLE:-1}"
GPU="${GPU:-${ROCR_VISIBLE_DEVICES%%,*}}"; GPU="${GPU:-0}"
PORT="${PORT:-29773}"
CAP="${SEG_SECONDS:-6600}"   # graceful stop before the 2h SLURM wall (resume next segment)
RUNDIR="$OUT/$RUN"
LAST_HOST="$RUNDIR/checkpoints/last/pretrained_model"
LOG="$OUT/${RUN}.$(date +%s).log"
DONE=$(printf '%06d' "$STEPS")

[ -f "$OUT/run_train_qat_student.py" ] || { echo "[qat-ft] missing /outputs/run_train_qat_student.py"; exit 3; }
[ -f "$PKG/quant/quant_student.py" ]   || { echo "[qat-ft] missing $PKG/quant/quant_student.py"; exit 3; }
[ -d "$PYLIBS/brevitas" ] || [ -d "$PYLIBS" ] || { echo "[qat-ft] missing pylibs $PYLIBS"; exit 3; }
if [ -d "$RUNDIR/checkpoints/$DONE" ]; then
  echo "[qat-ft] $RUN already at $STEPS steps -> skip"; exit 0
fi

COMMON="--policy.type=molmoact2 --policy.checkpoint_path=allenai/MolmoAct2-LIBERO --policy.push_to_hub=false \
  --policy.enable_lora_vlm=true --policy.action_mode=both \
  --policy.chunk_size=10 --policy.n_action_steps=10 --policy.model_dtype=bfloat16 \
  --policy.num_flow_timesteps=8 --policy.gradient_checkpointing=true \
  --policy.roi_prune_enable=false --policy.roi_fastv_enable=false \
  --policy.roi_unfreeze_last_decoder=$UNFREEZE \
  --dataset.repo_id=allenai/MolmoAct2-LIBERO-Dataset --dataset.revision=main --dataset.video_backend=pyav \
  --batch_size=$BS --steps=$STEPS --save_freq=$SAVE_FREQ --log_freq=25"

if [ -f "$LAST_HOST/train_config.json" ]; then
  echo "[qat-ft] RESUME $RUN (variant=$VARIANT gpu=$GPU port=$PORT steps=$STEPS)"
  TRAIN="/outputs/run_train_qat_student.py --config_path=/outputs/$RUN/checkpoints/last/pretrained_model/train_config.json --resume=true"
else
  echo "[qat-ft] FRESH  $RUN (variant=$VARIANT gpu=$GPU port=$PORT steps=$STEPS)"
  rm -rf "$RUNDIR" 2>/dev/null || true
  TRAIN="/outputs/run_train_qat_student.py $COMMON --output_dir=/outputs/$RUN"
fi

echo "[qat-ft] host=$(hostname) img=$IMG variant=$VARIANT W${WEIGHT_BITS}A${ACT_BITS} trainable=$QAT_STUDENT_TRAINABLE"
echo "[qat-ft] fp32=$FP32 qat=$QAT cap=${CAP}s log=$LOG t=$(date -Is)"
timeout --signal=TERM "${CAP}" docker run --rm ${EXTRA_DOCKER:-} \
  --user "$(id -u):$(id -g)" \
  --device=/dev/kfd --device=/dev/dri --group-add video --group-add render \
  --ipc=host --shm-size=64g --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -e USER="$(id -un)" -e LOGNAME="$(id -un)" \
  -e HOME=/cache -e HF_HOME=/cache -e XDG_CACHE_HOME=/cache \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e HF_HUB_DOWNLOAD_TIMEOUT=120 -e TORCH_BLAS_PREFER_HIPBLASLT=0 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -e HIP_VISIBLE_DEVICES=$GPU \
  -e PYTHONPATH=/pylibs:/pkg \
  -e QUANT_VARIANT="$VARIANT" -e FP32_STUDENT="$FP32" -e QAT_STATE="$QAT" \
  -e WEIGHT_BITS="$WEIGHT_BITS" -e ACT_BITS="$ACT_BITS" \
  -e QAT_STUDENT_TRAINABLE="$QAT_STUDENT_TRAINABLE" \
  -v "$CACHE":/cache -v "$OUT":/outputs \
  -v "$PYLIBS":/pylibs -v "$PKG":/pkg \
  -v "$OV/roi_prune.py":"$MP/roi_prune.py":ro \
  -v "$OV/modeling_molmoact2.py":"$MP/modeling_molmoact2.py":ro \
  -v "$OV/configuration_molmoact2.py":"$MP/configuration_molmoact2.py":ro \
  -v "$OV/hf_modeling_molmoact2.py":"$MP/molmoact2_hf_model/modeling_molmoact2.py":ro \
  "$IMG" \
  bash -lc "accelerate launch --num_processes=1 --mixed_precision=bf16 --main_process_port=$PORT $TRAIN" > "$LOG" 2>&1
RC=$?
echo "[qat-ft] SEGMENT_DONE $RUN rc=$RC (124=cap-reached, resume next segment) t=$(date -Is)"
[ "$RC" = "124" ] && exit 0
exit "$RC"
