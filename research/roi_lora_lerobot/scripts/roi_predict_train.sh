#!/usr/bin/env bash
# One CAUSAL PREDICTIVE GATING training SEGMENT (single GPU). Identical two-stage
# pipeline as roi_fastv_train.sh (scatter-back mid-ViT seam=6 student gate + in-LLM
# FastV cut at L0), EXCEPT the gate that selects the ViT keep-set is scored on the
# PAST frame (roi_predict_horizon steps earlier) and supervised by the action-expert
# attention teacher on the CURRENT frame. At inference the gate predicted at step t-H
# drives the single-pass prune of step t -- strictly causal, teacher-free.
#
# Resume-aware + wall-clock capped (chain 2h segments via checkpoint resume). Overlay
# .py files (incl. the temporal-split processor) are bind-mounted (no image rebuild).
set -u
WORK="${WORK:-$HOME/molmoact2}"
CACHE="${CACHE:-$WORK/hf_cache}"
OUT="${OUT:-$WORK/outputs}"
# Default overlay = the committed repo overlay (nested layout; includes the temporal
# -split processor needed by gate_predict). Override OV for a WIP overlay.
OV="${OV:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../docker/roi_overlay" && pwd)}"
MP=/opt/lerobot/src/lerobot/policies/molmoact2
IMG="${IMG:-molmoact2-lerobot-train:rocm942}"
SEL="${SEL:-gate_predict}"
KEEP="${ROI_KEEP:-0.50}"              # patch keep_frac fed to the ViT (scatter-back)
FASTV_KEEP="${FASTV_KEEP:-0.50}"     # image-token keep target after the in-LLM cut
FASTV_L0="${FASTV_L0:-3}"            # #decoder layers before the FastV cut
UNFREEZE="${UNFREEZE:-4}"           # last N decoder layers full-FT
CURR_STEPS="${CURR_STEPS:-4000}"    # anneal keep 0.9 -> FASTV_KEEP over N forwards
CURR_START="${CURR_START:-0.9}"
H="${H:-10}"                         # roi_predict_horizon (= n_action_steps replan stride)
STEPS="${STEPS:-12000}"
GPU="${GPU:-0}"
PORT="${PORT:-29781}"
CAP="${SEG_SECONDS:-6600}"
TAG=$(python3 -c "print('keep%03d_fv%03d' % (int(round(${KEEP}*100)), int(round(${FASTV_KEEP}*100))))")
RUN="roi_predict_${TAG}_l0${FASTV_L0}${RUN_SUFFIX:-}"
RUNDIR="$OUT/$RUN"
LAST_HOST="$RUNDIR/checkpoints/last/pretrained_model"
LOG="$OUT/${RUN}.$(date +%s).log"
DONE=$(printf '%06d' "$STEPS")

if [ -d "$RUNDIR/checkpoints/$DONE" ]; then
  echo "[predict] $RUN already at $STEPS steps -> skip"; exit 0
fi

COMMON="--policy.type=molmoact2 --policy.checkpoint_path=allenai/MolmoAct2-LIBERO --policy.push_to_hub=false \
  --policy.enable_lora_vlm=true --policy.action_mode=both \
  --policy.chunk_size=10 --policy.n_action_steps=10 --policy.model_dtype=bfloat16 \
  --policy.num_flow_timesteps=8 --policy.gradient_checkpointing=true \
  --policy.roi_prune_enable=true --policy.roi_prune_select=$SEL --policy.roi_prune_keep_frac=$KEEP --policy.roi_prune_placeholder=mask \
  --policy.roi_prune_group_drop=false \
  --policy.roi_predict_horizon=$H \
  --policy.roi_fastv_enable=true --policy.roi_fastv_keep_frac=$FASTV_KEEP --policy.roi_fastv_layer=$FASTV_L0 \
  --policy.roi_unfreeze_last_decoder=$UNFREEZE \
  --policy.roi_curriculum_steps=$CURR_STEPS --policy.roi_curriculum_start=$CURR_START \
  --policy.roi_prune_gate_lr=${GATE_LR:-1e-2} --policy.roi_distill_weight=${DW:-20} --policy.roi_prune_gate_seam=${SEAM:-6} \
  --dataset.repo_id=allenai/MolmoAct2-LIBERO-Dataset --dataset.revision=main --dataset.video_backend=pyav \
  --batch_size=${BS:-8} --steps=$STEPS --save_freq=${SAVE_FREQ:-1000} --log_freq=25"

if [ -f "$LAST_HOST/train_config.json" ]; then
  echo "[predict] RESUME $RUN from last checkpoint (gpu=$GPU port=$PORT cap=${CAP}s steps=$STEPS H=$H)"
  TRAIN="-m lerobot.scripts.lerobot_train --config_path=/outputs/$RUN/checkpoints/last/pretrained_model/train_config.json --resume=true"
else
  echo "[predict] FRESH  $RUN (gpu=$GPU port=$PORT cap=${CAP}s steps=$STEPS H=$H sel=$SEL keep=$KEEP fastv=$FASTV_KEEP L0=$FASTV_L0)"
  rm -rf "$RUNDIR" 2>/dev/null || true
  TRAIN="-m lerobot.scripts.lerobot_train $COMMON --output_dir=/outputs/$RUN"
fi

timeout --signal=TERM "${CAP}" docker run --rm \
  --user "$(id -u):$(id -g)" \
  --device=/dev/kfd --device=/dev/dri --group-add video --group-add render \
  --ipc=host --shm-size=64g --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -e USER="$(id -un)" -e LOGNAME="$(id -un)" \
  -e HOME=/cache -e HF_HOME=/cache -e XDG_CACHE_HOME=/cache \
  -e HF_HUB_DOWNLOAD_TIMEOUT=120 -e TORCH_BLAS_PREFER_HIPBLASLT=0 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -e HIP_VISIBLE_DEVICES=$GPU \
  -e ROI_PRUNE_DEBUG=${ROI_PRUNE_DEBUG:-} \
  -e ROI_DUMP_AGREEMENT=${ROI_DUMP_AGREEMENT:-} \
  -v "$CACHE":/cache -v "$OUT":/outputs \
  -v "$OV/roi_prune.py":"$MP/roi_prune.py":ro \
  -v "$OV/modeling_molmoact2.py":"$MP/modeling_molmoact2.py":ro \
  -v "$OV/configuration_molmoact2.py":"$MP/configuration_molmoact2.py":ro \
  -v "$OV/molmoact2_hf_model/modeling_molmoact2.py":"$MP/molmoact2_hf_model/modeling_molmoact2.py":ro \
  -v "$OV/processor_molmoact2.py":"$MP/processor_molmoact2.py":ro \
  "$IMG" \
  bash -lc "accelerate launch --num_processes=1 --mixed_precision=bf16 --main_process_port=$PORT $TRAIN" > "$LOG" 2>&1
RC=$?
echo "[predict] SEGMENT_DONE $RUN rc=$RC (124=timeout/cap-reached, resume next segment)"
if [ "$RC" = "124" ]; then exit 0; fi
exit "$RC"
