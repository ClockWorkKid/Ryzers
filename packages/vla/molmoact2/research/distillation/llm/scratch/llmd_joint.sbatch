#!/usr/bin/env bash
#SBATCH --job-name=llmd_joint
#SBATCH -p mi325x
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -c 16
#SBATCH --mem=220G
#SBATCH -t 02:00:00
#SBATCH --output=/shared_nobackup/%u/molmoact2/llmd_joint_%x_%j.log
# Run D joint stage: distilled ViT (hybrid_droid, swapped in + trainable) +
# distilled thin-twin LLM (Stage-1) + KV adapters, jointly trained on LIBERO
# (flow-matching task loss + multi-level LLM distillation vs the frozen teacher)
# PLUS a DROID seam-retention distillation term on real frames each step
# (mirrors the run B hybrid_droid mixing). Frozen: teacher backbone + action
# expert. Apptainer + eval SIF. Graceful-capped segment; resumes from `last`
# so an afterany chain finishes the full run.
#   SMOKE=1 -> 30-step forward/loss validation (rc=0, finite loss) then exit.
set -u
U=$USER
WORK=/shared_nobackup/$U/molmoact2
CACHE=$WORK/hf_cache
OUT=$WORK/outputs
SIF=$WORK/images/molmoact2-lerobot-eval.sif
OV=$HOME/roi_lerobot_overlay
MP=/opt/lerobot/src/lerobot/policies/molmoact2

RUN="${RUN:-llmd_joint}"
STCKPT="${STCKPT:-/outputs/llm_distill/stage1/student_final.pt}"
VIT_CKPT="${VIT_CKPT:-/outputs/hybrid_droid_full.pt}"
VIT_KW="${VIT_KW:-{\"variant\":\"hybrid\",\"dim\":256,\"num_conv_blocks\":4,\"num_attn_blocks\":3,\"attn_heads\":8}}"
KD_W="${KD_W:-0.5}"
DROID_W="${DROID_W:-1.0}"
DROID_ROOT="${DROID_ROOT:-/cache/droid_frames}"
DROID_BS="${DROID_BS:-4}"
BS="${BS:-4}"
PORT="${PORT:-29792}"
SMOKE="${SMOKE:-0}"
# isolate the smoke run dir so a 30-step smoke never pollutes/gets resumed by the full run
[ "$SMOKE" = "1" ] && RUN="${RUN}_smoke"
if [ "$SMOKE" = "1" ]; then STEPS="${STEPS:-30}"; SAVE_FREQ="${SAVE_FREQ:-1000}"; CAP="${SEG_SECONDS:-2400}"
else STEPS="${STEPS:-24000}"; SAVE_FREQ="${SAVE_FREQ:-2000}"; CAP="${SEG_SECONDS:-6600}"; fi

GPU=$(echo "${ROCR_VISIBLE_DEVICES:-0}" | awk -F, '{print $1}')
RUNDIR=$OUT/$RUN
LAST=$RUNDIR/checkpoints/last/pretrained_model
LOG=$OUT/${RUN}.$(date +%s).log
DONE=$(printf '%06d' "$STEPS")

# stage launcher + student + droid loader + vit student module into the bind target
mkdir -p "$OUT/llm_distill"
for f in run_train_stage2.py student.py droid_data.py; do
  [ -f "$HOME/llm_distill/$f" ] && { cp -f "$HOME/llm_distill/$f" "$OUT/llm_distill/$f"; sed -i 's/\r$//' "$OUT/llm_distill/$f"; }
done
[ -f "$HOME/vit_student.py" ] && { cp -f "$HOME/vit_student.py" "$OUT/vit_student.py"; sed -i 's/\r$//' "$OUT/vit_student.py"; }

echo "[llmd_joint] host=$(hostname) job=${SLURM_JOB_ID:-none} run=$RUN smoke=$SMOKE steps=$STEPS bs=$BS kd_w=$KD_W droid_w=$DROID_W vit=$VIT_CKPT gpu=$GPU t=$(date -Is)"
[ -f "$SIF" ] || { echo "[llmd_joint] SIF missing"; exit 2; }
[ -f "$OUT/${STCKPT#/outputs/}" ] || { echo "[llmd_joint] ERROR stage1 ckpt missing: $STCKPT"; exit 4; }
[ -f "$OUT/${VIT_CKPT#/outputs/}" ] || { echo "[llmd_joint] ERROR distilled ViT missing: $VIT_CKPT"; exit 4; }
[ -f "$OUT/vit_student.py" ] || { echo "[llmd_joint] missing vit_student.py"; exit 3; }
[ -f "$OUT/llm_distill/run_train_stage2.py" ] || { echo "[llmd_joint] missing launcher"; exit 3; }
[ -f "$OUT/llm_distill/droid_data.py" ] || { echo "[llmd_joint] missing droid_data.py"; exit 3; }
[ -d "$CACHE/droid_frames" ] || { echo "[llmd_joint] ERROR DROID frames missing: $CACHE/droid_frames"; exit 4; }
if [ "$SMOKE" != "1" ] && [ -d "$RUNDIR/checkpoints/$DONE" ]; then echo "[llmd_joint] $RUN already at $STEPS -> skip"; exit 0; fi

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
  echo "[llmd_joint] RESUME $RUN"
  TRAIN="/outputs/llm_distill/run_train_stage2.py --config_path=/outputs/$RUN/checkpoints/last/pretrained_model/train_config.json --resume=true"
else
  echo "[llmd_joint] FRESH  $RUN"
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
  --bind "$OUT":/outputs \
  --bind "$OV/roi_prune.py":"$MP/roi_prune.py":ro \
  --bind "$OV/modeling_molmoact2.py":"$MP/modeling_molmoact2.py":ro \
  --bind "$OV/configuration_molmoact2.py":"$MP/configuration_molmoact2.py":ro \
  --bind "$OV/hf_modeling_molmoact2.py":"$MP/molmoact2_hf_model/modeling_molmoact2.py":ro \
  "$SIF" bash -lc "accelerate launch --num_processes=1 --mixed_precision=bf16 --main_process_port=$PORT $TRAIN" > "$LOG" 2>&1
RC=$?
echo "[llmd_joint] SEGMENT_DONE $RUN rc=$RC (124=cap-reached, resume next segment) log=$LOG t=$(date -Is)"
[ "$RC" = "124" ] && exit 0
exit "$RC"
