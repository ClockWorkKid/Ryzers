#!/usr/bin/env bash
# One finetune run of the distilled student ViT + LoRA(VLM) + action expert on
# LIBERO, via Docker + the training image (apptainer userns is broken on
# gpu-node; docker is the proven MI300X training path). Fresh-to-completion
# (resume if a `last` checkpoint already exists). ROI pruning is OFF (full
# student, distillation-recovery finetune). Runs stock lerobot_train through the
# run_train_student.py launcher so the student is a trainable submodule.
set -u
U=$USER
WORK=/shared_nobackup/$U/molmoact2
CACHE=$WORK/hf_cache
OUT=$WORK/outputs
OV=$HOME/roi_lerobot_overlay
MP=/opt/lerobot/src/lerobot/policies/molmoact2
IMG="${IMG:-molmoact2-lerobot-train:rocm942}"

RUN="${RUN:-student_ft_lora}"
STEPS="${STEPS:-12000}"
SAVE_FREQ="${SAVE_FREQ:-1000}"
BS="${BS:-8}"
UNFREEZE="${UNFREEZE:-4}"
GPU="${GPU:-${ROCR_VISIBLE_DEVICES%%,*}}"; GPU="${GPU:-0}"
PORT="${PORT:-29771}"
CAP="${SEG_SECONDS:-6600}"   # graceful stop before the 2h SLURM wall (resume next segment)
RUNDIR="$OUT/$RUN"
LAST_HOST="$RUNDIR/checkpoints/last/pretrained_model"
LOG="$OUT/${RUN}.$(date +%s).log"
DONE=$(printf '%06d' "$STEPS")

[ -f "$OUT/run_train_student.py" ] || { echo "[student-ft] missing /outputs/run_train_student.py"; exit 3; }
[ -f "$OUT/vit_student.py" ] || { echo "[student-ft] missing /outputs/vit_student.py"; exit 3; }
[ -f "$OUT/hybrid_full.pt" ] || { echo "[student-ft] missing /outputs/hybrid_full.pt"; exit 3; }
if [ -d "$RUNDIR/checkpoints/$DONE" ]; then
  echo "[student-ft] $RUN already at $STEPS steps -> skip"; exit 0
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
  echo "[student-ft] RESUME $RUN (gpu=$GPU port=$PORT steps=$STEPS)"
  TRAIN="/outputs/run_train_student.py --config_path=/outputs/$RUN/checkpoints/last/pretrained_model/train_config.json --resume=true"
else
  echo "[student-ft] FRESH  $RUN (gpu=$GPU port=$PORT steps=$STEPS)"
  rm -rf "$RUNDIR" 2>/dev/null || true
  TRAIN="/outputs/run_train_student.py $COMMON --output_dir=/outputs/$RUN"
fi

echo "[student-ft] host=$(hostname) img=$IMG cap=${CAP}s log=$LOG t=$(date -Is)"
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
  -e VIT_STUDENT_CKPT=/outputs/hybrid_full.pt \
  -e VIT_STUDENT_MODULE=/outputs/vit_student.py \
  -v "$CACHE":/cache -v "$OUT":/outputs \
  -v "$OV/roi_prune.py":"$MP/roi_prune.py":ro \
  -v "$OV/modeling_molmoact2.py":"$MP/modeling_molmoact2.py":ro \
  -v "$OV/configuration_molmoact2.py":"$MP/configuration_molmoact2.py":ro \
  -v "$OV/hf_modeling_molmoact2.py":"$MP/molmoact2_hf_model/modeling_molmoact2.py":ro \
  "$IMG" \
  bash -lc "accelerate launch --num_processes=1 --mixed_precision=bf16 --main_process_port=$PORT $TRAIN" > "$LOG" 2>&1
RC=$?
echo "[student-ft] SEGMENT_DONE $RUN rc=$RC (124=cap-reached, resume next segment) t=$(date -Is)"
[ "$RC" = "124" ] && exit 0
exit "$RC"
