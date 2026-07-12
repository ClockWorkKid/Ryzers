#!/usr/bin/env bash
# One redesign-v2 training SEGMENT, single GPU. Rollback to scatter-back patch-drop
# (ViT compute saved, FULL pooled grid still enters the LLM -> no OOD collapse) PLUS
# FastV in-LLM pruning (after L0 decoder layers the low-importance image tokens are
# dropped so the deep VLM layers AND the action-expert cross-attention run on the
# curriculum-annealed keep fraction) PLUS last-N decoder layers unfrozen for capacity.
#
# Resume-aware + wall-clock capped to fit the 2h partition; chain segments via
# checkpoint resume. Overlay .py files are bind-mounted (no image rebuild after edits).
set -u
WORK="${WORK:-$HOME/molmoact2}"
CACHE="${CACHE:-$WORK/hf_cache}"
OUT="${OUT:-$WORK/outputs}"
# Default overlay = the committed repo overlay (nested layout) next to this script, so
# `OV=<repo>/docker/roi_overlay` is picked up automatically. Override OV to bind a
# work-in-progress overlay instead.
OV="${OV:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../docker/roi_overlay" && pwd)}"
MP=/opt/lerobot/src/lerobot/policies/molmoact2
IMG="${IMG:-molmoact2-lerobot-train:rocm942}"
SEL="${SEL:-gate_distill}"
KEEP="${ROI_KEEP:-0.50}"              # patch keep_frac fed to the ViT (scatter-back)
FASTV="${FASTV:-true}"               # true = two-stage (Stage-1 gate + Stage-2 FastV); false = Stage-1 gate only
FASTV_KEEP="${FASTV_KEEP:-0.25}"     # image-token keep target after the in-LLM cut
FASTV_L0="${FASTV_L0:-3}"            # #decoder layers before the FastV cut
UNFREEZE="${UNFREEZE:-4}"           # last N decoder layers full-FT
CURR_STEPS="${CURR_STEPS:-4000}"    # anneal keep 0.9 -> FASTV_KEEP over N forwards (0=off)
CURR_START="${CURR_START:-0.9}"
STEPS="${STEPS:-12000}"
GPU="${GPU:-0}"
PORT="${PORT:-29763}"
CAP="${SEG_SECONDS:-6600}"
# Stage-2 FastV flags only when FASTV=true; otherwise this is the Stage-1 seam gate run
# (scatter-back, LLM sees full grid) -- e.g. the original seam-6 gate_distill experiment.
if [ "$FASTV" = "true" ]; then
  FASTV_FRAG="--policy.roi_fastv_enable=true --policy.roi_fastv_keep_frac=$FASTV_KEEP --policy.roi_fastv_layer=$FASTV_L0 \
  --policy.roi_curriculum_steps=$CURR_STEPS --policy.roi_curriculum_start=$CURR_START"
  TAG=$(python3 -c "print('keep%03d_fv%03d' % (int(round(${KEEP}*100)), int(round(${FASTV_KEEP}*100))))")
  RUN="roi_fastv_${TAG}"
else
  FASTV_FRAG=""
  TAG=$(python3 -c "print('keep%03d' % int(round(${KEEP}*100)))")
  RUN="roi_gate_${TAG}"
fi
RUN="${RUN}${RUN_SUFFIX:-}"
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
  --policy.roi_prune_group_drop=false \
  $FASTV_FRAG \
  --policy.roi_unfreeze_last_decoder=$UNFREEZE \
  --policy.roi_prune_gate_lr=${GATE_LR:-1e-2} --policy.roi_distill_weight=${DW:-20} --policy.roi_prune_gate_seam=${SEAM:-6} \
  --dataset.repo_id=allenai/MolmoAct2-LIBERO-Dataset --dataset.revision=main --dataset.video_backend=pyav \
  --batch_size=${BS:-8} --steps=$STEPS --save_freq=${SAVE_FREQ:-1000} --log_freq=25"

if [ -f "$LAST_HOST/train_config.json" ]; then
  echo "[train] RESUME $RUN from last checkpoint (gpu=$GPU port=$PORT cap=${CAP}s steps=$STEPS)"
  TRAIN="-m lerobot.scripts.lerobot_train --config_path=/outputs/$RUN/checkpoints/last/pretrained_model/train_config.json --resume=true"
else
  echo "[train] FRESH  $RUN (gpu=$GPU port=$PORT cap=${CAP}s steps=$STEPS)"
  rm -rf "$RUNDIR" 2>/dev/null || true
  TRAIN="-m lerobot.scripts.lerobot_train $COMMON --output_dir=/outputs/$RUN"
fi

timeout --signal=TERM "${CAP}" docker run --rm ${EXTRA_DOCKER:-} \
  --user "$(id -u):$(id -g)" \
  --device=/dev/kfd --device=/dev/dri --group-add video --group-add render \
  --ipc=host --shm-size=64g --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -e USER="$(id -un)" -e LOGNAME="$(id -un)" \
  -e HOME=/cache -e HF_HOME=/cache -e XDG_CACHE_HOME=/cache \
  -e HF_HUB_DOWNLOAD_TIMEOUT=120 -e TORCH_BLAS_PREFER_HIPBLASLT=${HIPBLASLT:-0} \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -e HIP_VISIBLE_DEVICES=$GPU \
  -e ROI_PRUNE_DEBUG=${ROI_PRUNE_DEBUG:-} \
  -v "$CACHE":/cache -v "$OUT":/outputs \
  -v "$OV/roi_prune.py":"$MP/roi_prune.py":ro \
  -v "$OV/modeling_molmoact2.py":"$MP/modeling_molmoact2.py":ro \
  -v "$OV/configuration_molmoact2.py":"$MP/configuration_molmoact2.py":ro \
  -v "$OV/molmoact2_hf_model/modeling_molmoact2.py":"$MP/molmoact2_hf_model/modeling_molmoact2.py":ro \
  "$IMG" \
  bash -lc "accelerate launch --num_processes=1 --mixed_precision=bf16 --main_process_port=$PORT $TRAIN" > "$LOG" 2>&1
RC=$?
echo "[train] SEGMENT_DONE $RUN rc=$RC (124=timeout/cap-reached, resume next segment)"
if [ "$RC" = "124" ]; then exit 0; fi
exit "$RC"
