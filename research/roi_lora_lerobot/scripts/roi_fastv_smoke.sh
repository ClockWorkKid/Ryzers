#!/usr/bin/env bash
# Fast correctness smoke for the redesign-v2 path: scatter-back patch-drop (ViT
# compute saved, FULL pooled grid still enters the LLM) + FastV in-LLM pruning
# (after L0 decoder layers the low-importance image tokens are dropped so deeper
# layers AND the action-expert cross-attention run on ~FASTV_KEEP of image tokens)
# + last-N decoder layers unfrozen. Validates: (1) overlay imports + builds the
# model, (2) the FastV mid-forward cut runs without shape errors (ROI_PRUNE_DEBUG
# prints "[roi-fastv] ... seq S->S_new"), (3) the loss is finite. NOT convergence.
set -u
WORK="${WORK:-$HOME/molmoact2}"
CACHE="${CACHE:-$WORK/hf_cache}"
OUT="${OUT:-$WORK/outputs}"
OV="${OV:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../docker/roi_overlay" && pwd)}"
MP=/opt/lerobot/src/lerobot/policies/molmoact2
IMG="${IMG:-molmoact2-lerobot-train:rocm942}"
KEEP="${ROI_KEEP:-0.50}"          # patch keep_frac fed to the ViT (scatter-back)
FASTV_KEEP="${FASTV_KEEP:-0.25}"  # image-token keep after the in-LLM cut (target)
FASTV_L0="${FASTV_L0:-3}"         # #decoder layers before the FastV cut
UNFREEZE="${UNFREEZE:-4}"         # last N decoder layers full-FT
GPU="${GPU:-0}"
PORT="${PORT:-29762}"
STEPS="${STEPS:-4}"
RUN="roi_fastv_smoke"
RUNDIR="$OUT/$RUN"
LOG="$OUT/${RUN}.$(date +%s).log"
rm -rf "$RUNDIR" 2>/dev/null || true

# curriculum OFF in the smoke (steps=0) so the cut fires at the target keep immediately.
COMMON="--policy.type=molmoact2 --policy.checkpoint_path=allenai/MolmoAct2-LIBERO --policy.push_to_hub=false \
  --policy.enable_lora_vlm=true --policy.action_mode=both \
  --policy.chunk_size=10 --policy.n_action_steps=10 --policy.model_dtype=bfloat16 \
  --policy.num_flow_timesteps=8 --policy.gradient_checkpointing=true \
  --policy.roi_prune_enable=true --policy.roi_prune_select=gate_distill --policy.roi_prune_keep_frac=$KEEP --policy.roi_prune_placeholder=mask \
  --policy.roi_prune_group_drop=false \
  --policy.roi_fastv_enable=true --policy.roi_fastv_keep_frac=$FASTV_KEEP --policy.roi_fastv_layer=$FASTV_L0 \
  --policy.roi_unfreeze_last_decoder=$UNFREEZE --policy.roi_curriculum_steps=0 \
  --policy.roi_prune_gate_lr=1e-2 --policy.roi_distill_weight=20 --policy.roi_prune_gate_seam=6 \
  --dataset.repo_id=allenai/MolmoAct2-LIBERO-Dataset --dataset.revision=main --dataset.video_backend=pyav \
  --batch_size=${BS:-2} --steps=$STEPS --save_freq=$STEPS --log_freq=1"

echo "[smoke] fastv keep=$KEEP fastv_keep=$FASTV_KEEP L0=$FASTV_L0 unfreeze=$UNFREEZE steps=$STEPS gpu=$GPU -> $LOG"
docker run --rm ${EXTRA_DOCKER:-} \
  --user "$(id -u):$(id -g)" \
  --device=/dev/kfd --device=/dev/dri --group-add video --group-add render \
  --ipc=host --shm-size=64g --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -e USER="$(id -un)" -e LOGNAME="$(id -un)" \
  -e HOME=/cache -e HF_HOME=/cache -e XDG_CACHE_HOME=/cache \
  -e HF_HUB_DOWNLOAD_TIMEOUT=120 -e TORCH_BLAS_PREFER_HIPBLASLT=${HIPBLASLT:-0} \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -e HIP_VISIBLE_DEVICES=$GPU -e ROI_PRUNE_DEBUG=1 \
  -v "$CACHE":/cache -v "$OUT":/outputs \
  -v "$OV/roi_prune.py":"$MP/roi_prune.py":ro \
  -v "$OV/modeling_molmoact2.py":"$MP/modeling_molmoact2.py":ro \
  -v "$OV/configuration_molmoact2.py":"$MP/configuration_molmoact2.py":ro \
  -v "$OV/molmoact2_hf_model/modeling_molmoact2.py":"$MP/molmoact2_hf_model/modeling_molmoact2.py":ro \
  "$IMG" \
  bash -lc "accelerate launch --num_processes=1 --mixed_precision=bf16 --main_process_port=$PORT -m lerobot.scripts.lerobot_train $COMMON --output_dir=/outputs/$RUN" > "$LOG" 2>&1
RC=$?
echo "[smoke] rc=$RC"
echo "===== fastv / roi debug lines ====="; grep -a "roi-fastv\|roi-prune\|roi-groupdrop\|unfroze" "$LOG" | head -20
echo "===== loss lines ====="; grep -aE "loss|flow|distill|self_distill|step:" "$LOG" | tail -20
echo "===== errors (if any) ====="; grep -aiE "error|traceback|assert|nan|inf" "$LOG" | head -30
