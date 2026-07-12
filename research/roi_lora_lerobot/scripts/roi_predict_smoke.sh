#!/usr/bin/env bash
# CAUSAL PREDICTIVE GATING correctness smoke (a few real training steps). Validates
# end-to-end on real LIBERO data (rule: validate the data path before full training):
#   (1) observation_delta_indices=[-H,0] loads a past frame; the processor splits it
#       into pixel_values(current) + past_pixel_values and the batch reaches forward,
#   (2) the gate is scored on the PAST frame (build_batched_images reshape ok),
#   (3) the predicted keep-set prunes the CURRENT ViT (ROI_PRUNE_DEBUG line prints),
#   (4) the masked predictive BCE loss computes with grad into the gate, and
#   (5) precision@K of the past-gate vs current-teacher beats the random baseline.
# Writes an agreement JSONL that the launcher inspects before committing full runs.
set -u
WORK="${WORK:-$HOME/molmoact2}"
CACHE="${CACHE:-$WORK/hf_cache}"
OUT="${OUT:-$WORK/outputs}"
OV="${OV:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../docker/roi_overlay" && pwd)}"
MP=/opt/lerobot/src/lerobot/policies/molmoact2
IMG="${IMG:-molmoact2-lerobot-train:rocm942}"
SEL="${SEL:-gate_predict}"
KEEP="${ROI_KEEP:-0.50}"
FASTV_KEEP="${FASTV_KEEP:-0.50}"
FASTV_L0="${FASTV_L0:-3}"
H="${H:-10}"
STEPS="${STEPS:-8}"
GPU="${GPU:-0}"
PORT="${PORT:-29785}"
RUN="roi_predict_smoke"
RUNDIR="$OUT/$RUN"
AGREE="$OUT/${RUN}_agreement.jsonl"
LOG="$OUT/${RUN}.$(date +%s).log"
rm -rf "$RUNDIR" "$AGREE" 2>/dev/null || true

TRAIN="-m lerobot.scripts.lerobot_train \
  --policy.type=molmoact2 --policy.checkpoint_path=allenai/MolmoAct2-LIBERO --policy.push_to_hub=false \
  --policy.enable_lora_vlm=true --policy.action_mode=both \
  --policy.chunk_size=10 --policy.n_action_steps=10 --policy.model_dtype=bfloat16 \
  --policy.num_flow_timesteps=8 --policy.gradient_checkpointing=true \
  --policy.roi_prune_enable=true --policy.roi_prune_select=$SEL --policy.roi_prune_keep_frac=$KEEP --policy.roi_prune_placeholder=mask \
  --policy.roi_prune_group_drop=false \
  --policy.roi_predict_horizon=$H \
  --policy.roi_fastv_enable=true --policy.roi_fastv_keep_frac=$FASTV_KEEP --policy.roi_fastv_layer=$FASTV_L0 \
  --policy.roi_unfreeze_last_decoder=4 \
  --policy.roi_curriculum_steps=0 --policy.roi_curriculum_start=$KEEP \
  --policy.roi_prune_gate_lr=1e-2 --policy.roi_distill_weight=20 --policy.roi_prune_gate_seam=6 \
  --dataset.repo_id=allenai/MolmoAct2-LIBERO-Dataset --dataset.revision=main --dataset.video_backend=pyav \
  --batch_size=${BS:-4} --steps=$STEPS --save_freq=100000 --log_freq=1 --output_dir=/outputs/$RUN"

echo "[predict-smoke] host=$(hostname) SEL=$SEL H=$H keep=$KEEP fastv=$FASTV_KEEP L0=$FASTV_L0 steps=$STEPS -> $LOG"
docker run --rm \
  --user "$(id -u):$(id -g)" \
  --device=/dev/kfd --device=/dev/dri --group-add video --group-add render \
  --ipc=host --shm-size=64g --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -e USER="$(id -un)" -e LOGNAME="$(id -un)" \
  -e HOME=/cache -e HF_HOME=/cache -e XDG_CACHE_HOME=/cache \
  -e HF_HUB_DOWNLOAD_TIMEOUT=120 -e TORCH_BLAS_PREFER_HIPBLASLT=0 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -e HIP_VISIBLE_DEVICES=$GPU \
  -e ROI_PRUNE_DEBUG=1 \
  -e ROI_DUMP_AGREEMENT=/outputs/${RUN}_agreement.jsonl \
  -v "$CACHE":/cache -v "$OUT":/outputs \
  -v "$OV/roi_prune.py":"$MP/roi_prune.py":ro \
  -v "$OV/modeling_molmoact2.py":"$MP/modeling_molmoact2.py":ro \
  -v "$OV/configuration_molmoact2.py":"$MP/configuration_molmoact2.py":ro \
  -v "$OV/molmoact2_hf_model/modeling_molmoact2.py":"$MP/molmoact2_hf_model/modeling_molmoact2.py":ro \
  -v "$OV/processor_molmoact2.py":"$MP/processor_molmoact2.py":ro \
  "$IMG" \
  bash -lc "accelerate launch --num_processes=1 --mixed_precision=bf16 --main_process_port=$PORT $TRAIN" > "$LOG" 2>&1
RC=$?
echo "[predict-smoke] rc=$RC"
echo "=== ROI prune debug + predict loss lines ==="
grep -E "roi-prune|roi_predict_loss|past_pixel|Traceback|Error|step:" "$LOG" | tail -30
echo "=== agreement records ==="
[ -f "$AGREE" ] && cat "$AGREE" || echo "NO AGREEMENT FILE (predict loss never fired)"
exit "$RC"
