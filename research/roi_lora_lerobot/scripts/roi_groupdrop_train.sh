#!/usr/bin/env bash
# One group-drop (pooling-aware) training SEGMENT, single GPU.
#
# Same distillation setup as the mid-ViT seam=6 gate_distill run (the cheap gate is
# distilled toward the frozen action-attention teacher), but with GROUP-DROP on:
# whole 2x2 pooling groups are pruned so the pooled/LLM token count drops ~linearly
# with keep_frac (real backbone prefill saving), instead of scatter-back (which keeps
# the token count fixed). This trains the LoRA adapter on the reduced-token space.
#
# Longer schedule than the seam=6 run (12000 steps) to check whether it converges
# better. Resume-aware + wall-clock capped to fit the 2h partition; chain segments
# via checkpoint resume (run_roi_train_detached.sh).
#
# Overlay .py files are bind-mounted, so no image rebuild is needed after edits.
set -u
CACHE=/shared_nobackup/user/molmoact2/hf_cache
OUT=/shared_nobackup/user/molmoact2/outputs
OV=$HOME/roi_lerobot_overlay
MP=/opt/lerobot/src/lerobot/policies/molmoact2
IMG=molmoact2-lerobot-train:rocm942
SEL="${SEL:-gate_distill}"          # gate distilled from the action-attention teacher
KEEP="${ROI_KEEP:?set ROI_KEEP (e.g. 0.25 0.50 0.75)}"
STEPS="${STEPS:-12000}"             # longer than the seam=6 run (6000)
GPU="${GPU:-0}"
PORT="${PORT:-29760}"
CAP="${SEG_SECONDS:-6600}"          # wall-clock cap per segment (< 2h partition limit)
TAG=$(python3 -c "print('keep%03d' % int(round(${KEEP}*100)))")
RUN="roi_groupdrop_${TAG}"
RUNDIR="$OUT/$RUN"
LAST_HOST="$RUNDIR/checkpoints/last/pretrained_model"
LOG="$OUT/${RUN}.$(date +%s).log"
DONE=$(printf '%06d' "$STEPS")

if [ -d "$RUNDIR/checkpoints/$DONE" ]; then
  echo "[train] $RUN already at $STEPS steps -> skip"; exit 0
fi

COMMON="--policy.type=molmoact2 --policy.checkpoint_path=allenai/MolmoAct2-LIBERO --policy.push_to_hub=false \
  --policy.enable_lora_vlm=true --policy.action_mode=both \
  --policy.chunk_size=10 --policy.n_action_steps=10 --policy.model_dtype=bfloat16 \
  --policy.num_flow_timesteps=8 --policy.gradient_checkpointing=true \
  --policy.roi_prune_enable=true --policy.roi_prune_select=$SEL --policy.roi_prune_keep_frac=$KEEP --policy.roi_prune_placeholder=mask \
  --policy.roi_prune_group_drop=true \
  --policy.roi_prune_gate_lr=${GATE_LR:-1e-2} --policy.roi_distill_weight=${DW:-20} --policy.roi_prune_gate_seam=${SEAM:-6} \
  --dataset.repo_id=allenai/MolmoAct2-LIBERO-Dataset --dataset.revision=main --dataset.video_backend=pyav \
  --batch_size=${BS:-8} --steps=$STEPS --save_freq=${SAVE_FREQ:-1000} --log_freq=25"

if [ -f "$LAST_HOST/train_config.json" ]; then
  echo "[train] RESUME $RUN from last checkpoint (gpu=$GPU port=$PORT cap=${CAP}s steps=$STEPS)"
  TRAIN="-m lerobot.scripts.lerobot_train --config_path=/outputs/$RUN/checkpoints/last/pretrained_model --resume=true"
else
  echo "[train] FRESH  $RUN (gpu=$GPU port=$PORT cap=${CAP}s steps=$STEPS)"
  rm -rf "$RUNDIR" 2>/dev/null || true
  TRAIN="-m lerobot.scripts.lerobot_train $COMMON --output_dir=/outputs/$RUN"
fi

timeout --signal=TERM "${CAP}" docker run --rm \
  --user "$(id -u):$(id -g)" \
  --device=/dev/kfd --device=/dev/dri --group-add video --group-add render \
  --ipc=host --shm-size=64g --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -e USER=user -e LOGNAME=user \
  -e HOME=/cache -e HF_HOME=/cache -e XDG_CACHE_HOME=/cache \
  -e HF_HUB_DOWNLOAD_TIMEOUT=120 -e TORCH_BLAS_PREFER_HIPBLASLT=0 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -e HIP_VISIBLE_DEVICES=$GPU \
  -e ROI_PRUNE_DEBUG=${ROI_PRUNE_DEBUG:-} \
  -v "$CACHE":/cache -v "$OUT":/outputs \
  -v "$OV/roi_prune.py":"$MP/roi_prune.py":ro \
  -v "$OV/modeling_molmoact2.py":"$MP/modeling_molmoact2.py":ro \
  -v "$OV/configuration_molmoact2.py":"$MP/configuration_molmoact2.py":ro \
  -v "$OV/hf_modeling_molmoact2.py":"$MP/molmoact2_hf_model/modeling_molmoact2.py":ro \
  "$IMG" \
  bash -lc "accelerate launch --num_processes=1 --mixed_precision=bf16 --main_process_port=$PORT $TRAIN" > "$LOG" 2>&1
RC=$?
echo "[train] SEGMENT_DONE $RUN rc=$RC (124=timeout/cap-reached, resume next segment)"
