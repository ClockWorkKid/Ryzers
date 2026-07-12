#!/usr/bin/env bash
#SBATCH --job-name=llmd_lora
#SBATCH -p mi325x
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -c 16
#SBATCH --mem=220G
#SBATCH -t 02:00:00
#SBATCH --output=/shared_nobackup/%u/molmoact2/llmd_lora_%x_%j.log
# Run D LoRA-adaptation stage (Phase 4): load the joint-distilled backbone
# (distilled ViT + thin-twin LLM + KV adapters) from the joint run, FREEZE it,
# and train only self-contained low-rank adapters on the student ViT + student
# LLM Linear leaves against the LIBERO flow-matching task loss (+ DROID seam
# retention). LLM-KD is off so the adapters can task-specialize. Apptainer + eval
# SIF. Graceful-capped segment; resumes from `last` so an afterany chain finishes.
#   SMOKE=1 -> 30-step forward/loss validation (rc=0, finite loss) then exit.
set -u
U=$USER
WORK=/shared_nobackup/$U/molmoact2
CACHE=$WORK/hf_cache
OUT=$WORK/outputs
SIF=$WORK/images/molmoact2-lerobot-eval.sif
OV=$HOME/roi_lerobot_overlay
MP=/opt/lerobot/src/lerobot/policies/molmoact2

RUN="${RUN:-llmd_lora}"
STCKPT="${STCKPT:-/outputs/llm_distill/stage1/student_final.pt}"
VIT_CKPT="${VIT_CKPT:-/outputs/hybrid_droid_full.pt}"
VIT_KW="${VIT_KW:-{\"variant\":\"hybrid\",\"dim\":256,\"num_conv_blocks\":4,\"num_attn_blocks\":3,\"attn_heads\":8}}"
JOINT_RUN="${JOINT_RUN:-llmd_joint}"
JOINT_CKPT="${JOINT_CKPT:-/outputs/$JOINT_RUN/checkpoints/last/pretrained_model}"
KD_W="${KD_W:-0.0}"
DROID_W="${DROID_W:-1.0}"
DROID_ROOT="${DROID_ROOT:-/cache/droid_frames}"
DROID_BS="${DROID_BS:-4}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
BS="${BS:-4}"
PORT="${PORT:-29793}"
SMOKE="${SMOKE:-0}"
# isolate the smoke run dir so a 30-step smoke never pollutes/gets resumed by the full run
[ "$SMOKE" = "1" ] && RUN="${RUN}_smoke"
if [ "$SMOKE" = "1" ]; then STEPS="${STEPS:-30}"; SAVE_FREQ="${SAVE_FREQ:-1000}"; CAP="${SEG_SECONDS:-2400}"
else STEPS="${STEPS:-10000}"; SAVE_FREQ="${SAVE_FREQ:-2000}"; CAP="${SEG_SECONDS:-6600}"; fi

GPU=$(echo "${ROCR_VISIBLE_DEVICES:-0}" | awk -F, '{print $1}')
RUNDIR=$OUT/$RUN
LAST=$RUNDIR/checkpoints/last/pretrained_model
LOG=$OUT/${RUN}.$(date +%s).log
DONE=$(printf '%06d' "$STEPS")

# stage launcher + student + lora + droid loader + vit student module
mkdir -p "$OUT/llm_distill"
for f in run_train_stage2.py student.py lora.py droid_data.py; do
  [ -f "$HOME/llm_distill/$f" ] && { cp -f "$HOME/llm_distill/$f" "$OUT/llm_distill/$f"; sed -i 's/\r$//' "$OUT/llm_distill/$f"; }
done
[ -f "$HOME/vit_student.py" ] && { cp -f "$HOME/vit_student.py" "$OUT/vit_student.py"; sed -i 's/\r$//' "$OUT/vit_student.py"; }

echo "[llmd_lora] host=$(hostname) job=${SLURM_JOB_ID:-none} run=$RUN smoke=$SMOKE steps=$STEPS bs=$BS r=$LORA_R alpha=$LORA_ALPHA kd_w=$KD_W droid_w=$DROID_W joint=$JOINT_CKPT gpu=$GPU t=$(date -Is)"
[ -f "$SIF" ] || { echo "[llmd_lora] SIF missing"; exit 2; }
[ -f "$OUT/${STCKPT#/outputs/}" ] || { echo "[llmd_lora] ERROR stage1 ckpt missing: $STCKPT"; exit 4; }
[ -f "$OUT/${VIT_CKPT#/outputs/}" ] || { echo "[llmd_lora] ERROR distilled ViT missing: $VIT_CKPT"; exit 4; }
[ -d "$OUT/${JOINT_CKPT#/outputs/}" ] || { echo "[llmd_lora] ERROR joint ckpt dir missing: $JOINT_CKPT"; exit 4; }
[ -f "$OUT/vit_student.py" ] || { echo "[llmd_lora] missing vit_student.py"; exit 3; }
[ -f "$OUT/llm_distill/run_train_stage2.py" ] || { echo "[llmd_lora] missing launcher"; exit 3; }
[ -f "$OUT/llm_distill/lora.py" ] || { echo "[llmd_lora] missing lora.py"; exit 3; }
[ -f "$OUT/llm_distill/droid_data.py" ] || { echo "[llmd_lora] missing droid_data.py"; exit 3; }
[ -d "$CACHE/droid_frames" ] || { echo "[llmd_lora] ERROR DROID frames missing: $CACHE/droid_frames"; exit 4; }
if [ "$SMOKE" != "1" ] && [ -d "$RUNDIR/checkpoints/$DONE" ]; then echo "[llmd_lora] $RUN already at $STEPS -> skip"; exit 0; fi

ulimit -n 65535 2>/dev/null || true

COMMON="--policy.type=molmoact2 --policy.checkpoint_path=allenai/MolmoAct2-LIBERO --policy.push_to_hub=false \
  --policy.enable_lora_vlm=false --policy.action_mode=both \
  --policy.chunk_size=10 --policy.n_action_steps=10 --policy.model_dtype=bfloat16 \
  --policy.num_flow_timesteps=8 --policy.gradient_checkpointing=true \
  --policy.roi_prune_enable=false --policy.roi_fastv_enable=false \
  --policy.roi_unfreeze_last_decoder=0 \
  --dataset.repo_id=allenai/MolmoAct2-LIBERO-Dataset --dataset.revision=main --dataset.video_backend=pyav \
  --batch_size=$BS --steps=$STEPS --save_freq=$SAVE_FREQ --log_freq=10"

if [ "$SMOKE" != "1" ] && [ -f "$LAST/train_config.json" ]; then
  echo "[llmd_lora] RESUME $RUN"
  TRAIN="/outputs/llm_distill/run_train_stage2.py --config_path=/outputs/$RUN/checkpoints/last/pretrained_model/train_config.json --resume=true"
else
  echo "[llmd_lora] FRESH  $RUN"
  rm -rf "$RUNDIR" 2>/dev/null || true
  TRAIN="/outputs/llm_distill/run_train_stage2.py $COMMON --output_dir=/outputs/$RUN"
fi

timeout --signal=TERM "$CAP" apptainer exec --rocm --cleanenv --home "$CACHE":/cache \
  --env USER="$U" --env LOGNAME="$U" \
  --env HF_HOME=/cache --env XDG_CACHE_HOME=/cache \
  --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
  --env HF_HUB_DOWNLOAD_TIMEOUT=120 --env TORCH_BLAS_PREFER_HIPBLASLT=0 \
  --env PYTORCH_ALLOC_CONF=expandable_segments:True \
  --env HIP_VISIBLE_DEVICES=$GPU \
  --env LLMD_STUDENT_CKPT="$STCKPT" --env LLMD_STUDENT_MODULE=/outputs/llm_distill/student.py \
  --env LLMD_KD_W="$KD_W" \
  --env VIT_STUDENT_CKPT="$VIT_CKPT" --env VIT_STUDENT_MODULE=/outputs/vit_student.py \
  --env VIT_STUDENT_KWARGS="$VIT_KW" \
  --env DROID_ROOT="$DROID_ROOT" --env DROID_MODULE=/outputs/llm_distill/droid_data.py \
  --env DROID_W="$DROID_W" --env DROID_BS="$DROID_BS" \
  --env LORA_STAGE=1 --env LORA_MODULE=/outputs/llm_distill/lora.py --env JOINT_CKPT="$JOINT_CKPT" \
  --env LORA_R="$LORA_R" --env LORA_ALPHA="$LORA_ALPHA" --env LORA_DROPOUT="$LORA_DROPOUT" \
  --bind "$OUT":/outputs \
  --bind "$OV/roi_prune.py":"$MP/roi_prune.py":ro \
  --bind "$OV/modeling_molmoact2.py":"$MP/modeling_molmoact2.py":ro \
  --bind "$OV/configuration_molmoact2.py":"$MP/configuration_molmoact2.py":ro \
  --bind "$OV/hf_modeling_molmoact2.py":"$MP/molmoact2_hf_model/modeling_molmoact2.py":ro \
  "$SIF" bash -lc "accelerate launch --num_processes=1 --mixed_precision=bf16 --main_process_port=$PORT $TRAIN" > "$LOG" 2>&1
RC=$?
echo "[llmd_lora] SEGMENT_DONE $RUN rc=$RC (124=cap-reached, resume next segment) log=$LOG t=$(date -Is)"
[ "$RC" = "124" ] && exit 0
exit "$RC"
