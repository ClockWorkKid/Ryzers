#!/usr/bin/env bash
# Fast correctness smoke for the group-drop training path (a handful of steps).
# Validates that: (1) the overlay edits import + build the model, (2) the teacher
# group selection + sequence shortening run without shape errors, (3) the LLM
# prefill actually shrinks (ROI_PRUNE_DEBUG prints "seq S->S_new"), and (4) the
# loss (flow + distill) is finite. NOT a convergence run.
set -u
CACHE=/shared_nobackup/user/molmoact2/hf_cache
OUT=/shared_nobackup/user/molmoact2/outputs
OV=$HOME/roi_lerobot_overlay
MP=/opt/lerobot/src/lerobot/policies/molmoact2
IMG=molmoact2-lerobot-train:rocm942
KEEP="${ROI_KEEP:-0.25}"
GPU="${GPU:-0}"
PORT="${PORT:-29761}"
STEPS="${STEPS:-4}"
RUN="roi_groupdrop_smoke"
RUNDIR="$OUT/$RUN"
LOG="$OUT/${RUN}.$(date +%s).log"
rm -rf "$RUNDIR" 2>/dev/null || true

COMMON="--policy.type=molmoact2 --policy.checkpoint_path=allenai/MolmoAct2-LIBERO --policy.push_to_hub=false \
  --policy.enable_lora_vlm=true --policy.action_mode=both \
  --policy.chunk_size=10 --policy.n_action_steps=10 --policy.model_dtype=bfloat16 \
  --policy.num_flow_timesteps=8 --policy.gradient_checkpointing=true \
  --policy.roi_prune_enable=true --policy.roi_prune_select=gate_distill --policy.roi_prune_keep_frac=$KEEP --policy.roi_prune_placeholder=mask \
  --policy.roi_prune_group_drop=true \
  --policy.roi_prune_gate_lr=1e-2 --policy.roi_distill_weight=20 --policy.roi_prune_gate_seam=6 \
  --dataset.repo_id=allenai/MolmoAct2-LIBERO-Dataset --dataset.revision=main --dataset.video_backend=pyav \
  --batch_size=${BS:-2} --steps=$STEPS --save_freq=$STEPS --log_freq=1"

echo "[smoke] group-drop keep=$KEEP steps=$STEPS gpu=$GPU -> $LOG"
docker run --rm \
  --user "$(id -u):$(id -g)" \
  --device=/dev/kfd --device=/dev/dri --group-add video --group-add render \
  --ipc=host --shm-size=64g --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -e USER=user -e LOGNAME=user \
  -e HOME=/cache -e HF_HOME=/cache -e XDG_CACHE_HOME=/cache \
  -e HF_HUB_DOWNLOAD_TIMEOUT=120 -e TORCH_BLAS_PREFER_HIPBLASLT=0 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -e HIP_VISIBLE_DEVICES=$GPU -e ROI_PRUNE_DEBUG=1 \
  -v "$CACHE":/cache -v "$OUT":/outputs \
  -v "$OV/roi_prune.py":"$MP/roi_prune.py":ro \
  -v "$OV/modeling_molmoact2.py":"$MP/modeling_molmoact2.py":ro \
  -v "$OV/configuration_molmoact2.py":"$MP/configuration_molmoact2.py":ro \
  -v "$OV/hf_modeling_molmoact2.py":"$MP/molmoact2_hf_model/modeling_molmoact2.py":ro \
  "$IMG" \
  bash -lc "accelerate launch --num_processes=1 --mixed_precision=bf16 --main_process_port=$PORT -m lerobot.scripts.lerobot_train $COMMON --output_dir=/outputs/$RUN" > "$LOG" 2>&1
RC=$?
echo "[smoke] rc=$RC"
echo "===== group-drop debug lines ====="; grep -a "roi-groupdrop\|roi-prune" "$LOG" | head -20
echo "===== loss lines ====="; grep -aE "loss|flow|distill|step:" "$LOG" | tail -20
echo "===== errors (if any) ====="; grep -aiE "error|traceback|assert|nan|inf" "$LOG" | head -30
