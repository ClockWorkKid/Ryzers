#!/usr/bin/env bash
#SBATCH --job-name=lp_curric
#SBATCH --output=%x_%j.log
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
# Single-node 8-GPU LIBERO-plus curriculum-LoRA training, ONE fixed-length SEGMENT per job.
# Self-resubmits (same node) until this stage reaches its STEPS target; on completion it submits
# the merge (which chains eval_k in parallel with train_(k+1)). This self-chaining fits clusters
# that enforce a per-job walltime cap and has no single "stage-done" job to hang an afterok on.
#
# Submit with:  sbatch -p "$GPU_PARTITION" --export=ALL,STAGE=1,LRANK=16,STEPS=34969,BS=8 train.sbatch
# Params via --export: STAGE, LRANK, CKPT, STEPS, BS, RETRY, GPU_PARTITION.
set -u
U="$USER"
SCR="${SCR:-$(cd "$(dirname "$0")" && pwd)}"          # dir holding train.sh + merge.sbatch
WORK="${WORK:-$HOME/molmoact2}"
OUT="${OUT:-$WORK/outputs}"
MERGEROOT="$OUT/lp_curric_merged"
IMG="${IMG:-molmoact2-lerobot-train:rocm950}"
BCTX="${BCTX:-$SCR/../docker}"                          # build context (Dockerfile.train)
GPU_PARTITION="${GPU_PARTITION:-}"
NODE="${SLURMD_NODENAME:-$(hostname)}"
RANKS=(16 32 64 128 256)
STAGE="${STAGE:-1}"; LRANK="${LRANK:-16}"; CKPT="${CKPT:-allenai/MolmoAct2-LIBERO}"
STEPS="${STEPS:-34969}"; BS="${BS:-8}"; RETRY="${RETRY:-0}"; MAXR="${MAXR:-15}"
SMOKE="${SMOKE:-0}"; LR_SCALE="${LR_SCALE:-1.0}"
RUN="${RUN:-lp_curric_s${STAGE}}"
for s in "$SCR"/*.sh; do [ -f "$s" ] && sed -i 's/\r$//' "$s" 2>/dev/null; done
echo "[sb] host=$NODE stage=$STAGE run=$RUN rank=$LRANK steps=$STEPS bs=$BS ckpt=$CKPT retry=$RETRY t=$(date -Is)"

# node-local image (build from the training Dockerfile if missing; one-time)
if ! docker image inspect "$IMG" >/dev/null 2>&1; then
  echo "[sb] image missing on $NODE -> building from $BCTX/Dockerfile.train"
  [ -f "$BCTX/Dockerfile.train" ] || { echo "[sb] no build ctx -> abort"; exit 3; }
  ( cd "$BCTX" && docker build -t "$IMG" -f Dockerfile.train . ) || { echo "[sb] BUILD FAIL"; exit 3; }
fi

export EXTRA_DOCKER="--name ${U}-lpcurric-s${STAGE}-${SLURM_JOB_ID:-$$}"
export IMG RUN STEPS BS LRANK LALPHA="$LRANK" CKPT SEG_SECONDS="${SEG_SECONDS:-13800}"
export WORK OUT DS_ROOT="${DS_ROOT:-}" LR_SCALE
T0=$SECONDS
bash "$SCR/train.sh"; RC=$?
ELAPSED=$((SECONDS - T0))

if [ "$SMOKE" = "1" ]; then
  echo "[sb] SMOKE mode -> no chaining. rc=$RC run=$RUN steps=$STEPS t=$(date -Is)"; exit "$RC"
fi

DONE=$(printf '%06d' "$STEPS")
LAST="$OUT/$RUN/checkpoints/last/pretrained_model"
PARG=""; [ -n "$GPU_PARTITION" ] && PARG="-p $GPU_PARTITION"
resub(){ cd "$SCR" && sbatch $PARG -w "$NODE" --export="ALL,STAGE=$STAGE,LRANK=$LRANK,CKPT=$CKPT,STEPS=$STEPS,BS=$BS,DS_ROOT=$DS_ROOT,LR_SCALE=${LR_SCALE:-1.0},GPU_PARTITION=$GPU_PARTITION,RETRY=$1" "$SCR/train.sbatch"; }

if [ -d "$OUT/$RUN/checkpoints/$DONE" ]; then
  echo "[sb] stage $STAGE reached target $DONE -> submit merge (chains eval_$STAGE || train_$((STAGE+1)))"
  cd "$SCR" && sbatch $PARG -w "$NODE" \
    --export="ALL,STAGE=$STAGE,SRC=$OUT/$RUN/checkpoints/last/pretrained_model,DST=$MERGEROOT/s${STAGE},TRAIN_NODE=$NODE,DS_ROOT=$DS_ROOT,BS=$BS,LR_SCALE=${LR_SCALE:-1.0},GPU_PARTITION=$GPU_PARTITION" \
    "$SCR/merge.sbatch"
elif [ "$RC" = "0" ] || [ "$RC" = "124" ] || [ "$RC" = "20" ]; then
  # No-op guard: a productive segment always runs many minutes (dataloader warmup alone is long).
  # A clean exit in <10min that did NOT reach the target means 0 real progress (e.g. resuming an
  # already-finished checkpoint) -> STOP to avoid a fast infinite resubmit loop that hogs the node.
  if [ "$ELAPSED" -lt 600 ]; then
    echo "[sb] STOP: clean segment (rc=$RC) exited in ${ELAPSED}s (<600s) with no progress to $DONE -> loop guard tripped (check RUN/CKPT/STEPS)"
  else
    echo "[sb] clean segment (rc=$RC, ${ELAPSED}s) not at target -> resubmit stage $STAGE on $NODE"
    resub 0
  fi
elif [ -f "$LAST/train_config.json" ] && [ "$RETRY" -lt "$MAXR" ]; then
  echo "[sb] crash rc=$RC but ckpt exists -> resume (retry $((RETRY+1))/$MAXR)"
  resub $((RETRY+1))
else
  echo "[sb] STOP stage=$STAGE rc=$RC retry=$RETRY (inspect log)"
fi
echo "[sb] segment end stage=$STAGE rc=$RC t=$(date -Is)"
