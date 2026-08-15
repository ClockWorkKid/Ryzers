#!/usr/bin/env bash
# Run ONE LIBERO-plus curriculum-LoRA training SEGMENT on a SINGLE GPU node (N GPUs) via
# docker + accelerate (num_processes=N, single machine -> no multi-node NCCL). Resume-aware:
# re-invoke to continue from checkpoints/last. Set GRAD_CKPT=false on GPUs without gradient
# checkpointing support.
#
# This is the single-machine entry point. On a SLURM cluster it is driven by train.sbatch,
# which chains fixed-length segments to fit a per-job walltime cap.
#
# Key env (all have defaults):
#   WORK       project root            (default: $HOME/molmoact2)
#   CACHE      HF cache (HF_HOME)       (default: $WORK/hf_cache)
#   OUT        checkpoint output root   (default: $WORK/outputs)
#   IMG        training docker image    (default: molmoact2-lerobot-train:rocm950)
#   DS         lerobot dataset repo id  (default: Sylvest/libero_plus_lerobot)
#   CKPT       warm-start checkpoint    (default: allenai/MolmoAct2-LIBERO, or a merged stage dir)
#   RUN        run/output dir name      (default: lp_curric_s1)
#   LRANK/LALPHA  LoRA rank / alpha     (curriculum: 16, 32, 64, 128, 256)
#   STEPS BS NPROC  train steps / per-GPU batch / GPUs
#   SEG_SECONDS  wall cap for this segment (0 = no cap; >0 => timeout+resume)
set -u
U="$USER"
WORK="${WORK:-$HOME/molmoact2}"
CACHE="${CACHE:-$WORK/hf_cache}"
OUT="${OUT:-$WORK/outputs}"
IMG="${IMG:-molmoact2-lerobot-train:rocm950}"
DS="${DS:-Sylvest/libero_plus_lerobot}"
DS_ROOT="${DS_ROOT:-}"              # optional explicit local dataset dir (in-container path)
CKPT="${CKPT:-allenai/MolmoAct2-LIBERO}"
RUN="${RUN:-lp_curric_s1}"
STEPS="${STEPS:-4}"
BS="${BS:-8}"                       # per-GPU; effective = NPROC*BS
NPROC="${NPROC:-8}"
CHUNK="${CHUNK:-10}"
SAVE_FREQ="${SAVE_FREQ:-2000}"
VLM_MODE="${VLM_MODE:-lora}"
LRANK="${LRANK:-16}"; LALPHA="${LALPHA:-16}"
LORA_FULL_MODEL="${LORA_FULL_MODEL:-false}"
GRAD_CKPT="${GRAD_CKPT:-true}"
AUG="${AUG:-true}"
NW="${NW:-6}"
MODEL_DTYPE="${MODEL_DTYPE:-bfloat16}"
MIXED_PREC="${MIXED_PREC:-bf16}"
PORT="${PORT:-29761}"
CAP="${SEG_SECONDS:-0}"             # 0 = no wall cap; >0 chains segments
RUNDIR="$OUT/$RUN"
LAST_HOST="$RUNDIR/checkpoints/last/pretrained_model"
LOG="$OUT/${RUN}.$(date +%s).log"
mkdir -p "$OUT"

# dataset guard (lerobot home = $CACHE/lerobot)
if ! find "$CACHE/lerobot" -path "*libero_plus_lerobot*/meta/info.json" 2>/dev/null | grep -q .; then
  echo "[train] dataset $DS not staged under $CACHE/lerobot -> run stage_data.py first (rc=20)"; exit 20
fi

if [ "$VLM_MODE" = "lora" ]; then
  CAP_ARGS="--policy.enable_lora_vlm=true --policy.lora_rank=$LRANK --policy.lora_alpha=$LALPHA"
  [ "$LORA_FULL_MODEL" = "true" ] && CAP_ARGS="$CAP_ARGS --policy.lora_full_model=true"
else
  CAP_ARGS="--policy.enable_lora_vlm=false"
fi

DS_ROOT_ARG=""; [ -n "$DS_ROOT" ] && DS_ROOT_ARG="--dataset.root=$DS_ROOT"

# CKPT path translation: the host $OUT is bind-mounted at /outputs inside the container, so a
# merged-checkpoint host path (e.g. $OUT/lp_curric_merged/s1) is invisible in-container and the
# MolmoAct2 loader wrongly falls back to snapshot_download. Remap it to the in-container mount.
# HF repo ids (namespace/name) don't match $OUT/* so they pass through unchanged.
case "$CKPT" in
  "$OUT"/*) CKPT="/outputs/${CKPT#"$OUT"/}"; echo "[train] CKPT remapped to container path: $CKPT";;
esac

# Optional LR scaling for large-batch training (linear/sqrt rule when effective batch grows).
# Base component LRs are the MolmoAct2-LIBERO recipe (effective batch 64). Applied only on a
# FRESH start; on resume the scaled LRs are already baked into the checkpoint train_config.json.
LR_SCALE="${LR_SCALE:-1.0}"
LR_ARGS=""
if [ "$LR_SCALE" != "1.0" ] && [ "$LR_SCALE" != "1" ]; then
  read -r L_LLM L_VIT L_CONN L_AE L_GATE <<EOF2
$(python3 -c "s=float('$LR_SCALE'); print(f'{1e-5*s:.6g} {5e-6*s:.6g} {5e-6*s:.6g} {5e-5*s:.6g} {5e-5*s:.6g}')")
EOF2
  LR_ARGS="--policy.optimizer_lr=$L_LLM --policy.optimizer_vit_lr=$L_VIT --policy.optimizer_connector_lr=$L_CONN --policy.optimizer_action_expert_lr=$L_AE --policy.roi_prune_gate_lr=$L_GATE"
  echo "[train] LR_SCALE=$LR_SCALE -> llm=$L_LLM vit=$L_VIT conn=$L_CONN ae=$L_AE gate=$L_GATE"
fi

COMMON="--dataset.repo_id=$DS $DS_ROOT_ARG --dataset.video_backend=pyav \
  --dataset.image_transforms.enable=$AUG \
  --policy.type=molmoact2 --policy.checkpoint_path=$CKPT --policy.device=cuda \
  $CAP_ARGS --policy.action_mode=both $LR_ARGS \
  --policy.chunk_size=$CHUNK --policy.n_action_steps=$CHUNK --policy.num_flow_timesteps=8 \
  --policy.setup_type='single franka robotic arm in libero' \
  --policy.control_mode='delta end-effector pose' \
  --policy.image_keys='[\"observation.images.front\",\"observation.images.wrist\"]' \
  --policy.model_dtype=$MODEL_DTYPE --policy.gradient_checkpointing=$GRAD_CKPT \
  --policy.freeze_embedding=true --policy.normalize_gripper=false --policy.push_to_hub=false \
  --num_workers=$NW --batch_size=$BS --steps=$STEPS \
  --save_checkpoint=true --save_freq=$SAVE_FREQ --log_freq=1 --wandb.enable=false"

if [ -f "$LAST_HOST/train_config.json" ]; then
  echo "[train] RESUME $RUN (${NPROC}-GPU bs=$BS eff=$((NPROC*BS)) rank=$LRANK steps=$STEPS)"
  TRAIN="-m lerobot.scripts.lerobot_train --config_path=/outputs/$RUN/checkpoints/last/pretrained_model/train_config.json --resume=true"
else
  echo "[train] FRESH  $RUN (${NPROC}-GPU bs=$BS eff=$((NPROC*BS)) rank=$LRANK steps=$STEPS ckpt=$CKPT)"
  rm -rf "$RUNDIR" 2>/dev/null || true
  TRAIN="-m lerobot.scripts.lerobot_train $COMMON --output_dir=/outputs/$RUN --job_name=$RUN"
fi

RUNNER="timeout --signal=TERM $CAP"; [ "$CAP" = "0" ] && RUNNER=""
$RUNNER docker run --rm ${EXTRA_DOCKER:-} \
  --user "$(id -u):$(id -g)" \
  --device=/dev/kfd --device=/dev/dri --group-add video --group-add render \
  --ipc=host --shm-size=64g --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -e USER="$(id -un)" -e LOGNAME="$(id -un)" \
  -e HOME=/cache -e HF_HOME=/cache -e XDG_CACHE_HOME=/cache -e HF_HUB_DOWNLOAD_TIMEOUT=120 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -e TORCH_BLAS_PREFER_HIPBLASLT="${HIPBLASLT:-0}" \
  -v "$CACHE":/cache -v "$OUT":/outputs \
  "$IMG" \
  bash -lc "accelerate launch --multi_gpu --num_processes=$NPROC --mixed_precision=$MIXED_PREC --dynamo_backend=no --main_process_port=$PORT $TRAIN" > "$LOG" 2>&1
RC=$?
echo "[train] SEGMENT_DONE $RUN rc=$RC (124=cap-reached) log=$LOG t=$(date -Is)"
tail -n 25 "$LOG" 2>/dev/null
[ "$RC" = "124" ] && exit 0
exit "$RC"
