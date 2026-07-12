#!/usr/bin/env bash
#SBATCH --job-name=sft210
#SBATCH -p mi210
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 04:00:00
#SBATCH --output=/shared_nobackup/%u/molmoact2/sft_%x_%j.log
# Finetune a distilled student (student_vit + LoRA(VLM) + action expert) on LIBERO
# via apptainer + eval SIF on mi210 (validated; mi300x docker path is contended).
# One graceful-capped 4h segment; resumes from `last`, so an afterany chain finishes
# the full 12k-step run. Architecture per variant is keyed off NAME (no JSON in
# --export). Auto-promotes the distilled .pt from the distill output dir.
#   sbatch --job-name=ft_siglip --export=ALL,NAME=siglip_nano student_ft_mi210.sbatch
set -u
U=$USER
WORK=/shared_nobackup/$U/molmoact2
CACHE=$WORK/hf_cache
OUT=$WORK/outputs
SIF=$WORK/images/molmoact2-lerobot-eval.sif
OV=$HOME/roi_lerobot_overlay
MP=/opt/lerobot/src/lerobot/policies/molmoact2
DISTILL_RES=$WORK/vit_distill/resource/ckpt/vit_distill

NAME="${NAME:-siglip_nano}"
RUN="${RUN:-ft_$NAME}"
STEPS="${STEPS:-12000}"
SAVE_FREQ="${SAVE_FREQ:-2000}"          # MUST be < steps-per-segment (~5150 @2.68s/step,3h50m)
                                        # else a segment never checkpoints and the chain restarts from 0
BS="${BS:-8}"
UNFREEZE="${UNFREEZE:-4}"
PORT="${PORT:-29780}"
CAP="${SEG_SECONDS:-13800}"              # ~3h50m graceful stop before the 4h wall

# variant -> distilled ckpt filename + student build kwargs (single-quoted = safe)
case "$NAME" in
  siglip_nano)  CKF=siglip_nano_full.pt;  STKW='{"variant":"tinyvit","dim":240,"num_attn_blocks":4,"attn_heads":8}' ;;
  hybrid_droid) CKF=hybrid_droid_full.pt; STKW='{"variant":"hybrid","dim":256,"num_conv_blocks":4,"num_attn_blocks":3,"attn_heads":8}' ;;
  cnn_fpga)     CKF=cnn_fpga_full.pt;     STKW='{"variant":"cnn","dim":384,"num_conv_blocks":16,"conv_kernel":3}' ;;
  hybrid)       CKF=hybrid_full.pt;       STKW='{}' ;;
  *) echo "[sft210] unknown NAME=$NAME"; exit 3 ;;
esac
STCKPT=/outputs/$CKF

GPU=$(echo "${ROCR_VISIBLE_DEVICES:-0}" | awk -F, '{print $1}')
RUNDIR=$OUT/$RUN
LAST=$RUNDIR/checkpoints/last/pretrained_model
LOG=$OUT/${RUN}.$(date +%s).log
DONE=$(printf '%06d' "$STEPS")

for f in run_train_student.py vit_student.py; do
  [ -f "$HOME/$f" ] && { cp -f "$HOME/$f" "$OUT/$f"; sed -i 's/\r$//' "$OUT/$f"; }
done
# promote distilled checkpoint into the bind target if needed
if [ ! -f "$OUT/$CKF" ] && [ -f "$DISTILL_RES/$CKF" ]; then
  cp -f "$DISTILL_RES/$CKF" "$OUT/$CKF"; echo "[sft210] promoted $CKF from distill output"
fi
echo "[sft210] host=$(hostname) job=${SLURM_JOB_ID:-none} run=$RUN name=$NAME stckpt=$STCKPT kw=$STKW gpu=$GPU t=$(date -Is)"
[ -f "$SIF" ] || { echo "[sft210] SIF missing"; exit 2; }
if [ ! -f "$OUT/$CKF" ]; then echo "[sft210] ERROR distilled ckpt missing: $OUT/$CKF"; exit 4; fi
if [ -d "$RUNDIR/checkpoints/$DONE" ]; then echo "[sft210] $RUN already at $STEPS -> skip"; exit 0; fi

COMMON="--policy.type=molmoact2 --policy.checkpoint_path=allenai/MolmoAct2-LIBERO --policy.push_to_hub=false \
  --policy.enable_lora_vlm=true --policy.action_mode=both \
  --policy.chunk_size=10 --policy.n_action_steps=10 --policy.model_dtype=bfloat16 \
  --policy.num_flow_timesteps=8 --policy.gradient_checkpointing=true \
  --policy.roi_prune_enable=false --policy.roi_fastv_enable=false \
  --policy.roi_unfreeze_last_decoder=$UNFREEZE \
  --dataset.repo_id=allenai/MolmoAct2-LIBERO-Dataset --dataset.revision=main --dataset.video_backend=pyav \
  --batch_size=$BS --steps=$STEPS --save_freq=$SAVE_FREQ --log_freq=25"

if [ -f "$LAST/train_config.json" ]; then
  echo "[sft210] RESUME $RUN"
  TRAIN="/outputs/run_train_student.py --config_path=/outputs/$RUN/checkpoints/last/pretrained_model/train_config.json --resume=true"
else
  echo "[sft210] FRESH  $RUN"
  rm -rf "$RUNDIR" 2>/dev/null || true
  TRAIN="/outputs/run_train_student.py $COMMON --output_dir=/outputs/$RUN"
fi

timeout --signal=TERM "$CAP" apptainer exec --rocm --cleanenv --home "$CACHE":/cache \
  --env USER="$U" --env LOGNAME="$U" \
  --env HF_HOME=/cache --env XDG_CACHE_HOME=/cache \
  --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
  --env HF_HUB_DOWNLOAD_TIMEOUT=120 --env TORCH_BLAS_PREFER_HIPBLASLT=0 \
  --env PYTORCH_ALLOC_CONF=expandable_segments:True \
  --env HIP_VISIBLE_DEVICES=$GPU \
  --env VIT_STUDENT_CKPT="$STCKPT" --env VIT_STUDENT_MODULE=/outputs/vit_student.py \
  --env VIT_STUDENT_KWARGS="$STKW" \
  --bind "$OUT":/outputs \
  --bind "$OV/roi_prune.py":"$MP/roi_prune.py":ro \
  --bind "$OV/modeling_molmoact2.py":"$MP/modeling_molmoact2.py":ro \
  --bind "$OV/configuration_molmoact2.py":"$MP/configuration_molmoact2.py":ro \
  --bind "$OV/hf_modeling_molmoact2.py":"$MP/molmoact2_hf_model/modeling_molmoact2.py":ro \
  "$SIF" bash -lc "accelerate launch --num_processes=1 --mixed_precision=bf16 --main_process_port=$PORT $TRAIN" > "$LOG" 2>&1
RC=$?
echo "[sft210] SEGMENT_DONE $RUN rc=$RC (124=cap-reached, resume next segment) log=$LOG t=$(date -Is)"
[ "$RC" = "124" ] && exit 0
exit "$RC"
