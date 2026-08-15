#!/usr/bin/env bash
#SBATCH --job-name=lp_merge
#SBATCH --output=%x_%j.log
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:40:00
# Fold a completed stage-k LoRA adapter into a plain-base checkpoint (merge_curriculum.py), then
# optionally chain the next curriculum stage's training. Quick (~minutes, 1 GPU).
# Params via --export: STAGE, SRC (stage-k pretrained_model dir), DST (merged output dir),
#                      TRAIN_NODE, BS, LR_SCALE, GPU_PARTITION. Set NOCHAIN=1 to merge only.
set -u
U="$USER"
SCR="${SCR:-$(cd "$(dirname "$0")" && pwd)}"
WORK="${WORK:-$HOME/molmoact2}"
CACHE="${CACHE:-$WORK/hf_cache}"
OUT="${OUT:-$WORK/outputs}"
MERGEROOT="$OUT/lp_curric_merged"
IMG="${IMG:-molmoact2-lerobot-train:rocm950}"
RANKS=(16 32 64 128 256)
STAGE="${STAGE:-1}"; BS="${BS:-8}"; TRAIN_NODE="${TRAIN_NODE:-$(hostname)}"
NOCHAIN="${NOCHAIN:-0}"; LR_SCALE="${LR_SCALE:-1.0}"; GPU_PARTITION="${GPU_PARTITION:-}"
SRC="${SRC:-$OUT/lp_curric_s${STAGE}/checkpoints/last/pretrained_model}"
DST="${DST:-$MERGEROOT/s${STAGE}}"
DS_ROOT="${DS_ROOT:-}"
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "[merge-sb] image $IMG missing on $(hostname) -> abort"; exit 3; }

# in-container path remap (host $OUT bind-mounted at /outputs)
c_src="/outputs/${SRC#"$OUT"/}"; c_dst="/outputs/${DST#"$OUT"/}"
mkdir -p "$DST"
docker run --rm --name "${U}-lpmerge-${SLURM_JOB_ID:-$$}" \
  --user "$(id -u):$(id -g)" \
  --device=/dev/kfd --device=/dev/dri --group-add video --group-add render \
  --ipc=host --shm-size=32g --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -e HOME=/cache -e HF_HOME=/cache -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e HIP_VISIBLE_DEVICES=0 -e SRC="$c_src" -e DST="$c_dst" \
  -v "$CACHE":/cache -v "$OUT":/outputs -v "$SCR":/work \
  "$IMG" python3 /work/merge_curriculum.py; RC=$?
echo "[merge-sb] merge rc=$RC t=$(date -Is)"
[ "$RC" != "0" ] && { echo "[merge-sb] merge failed -> stop chain (stage $STAGE)"; exit "$RC"; }

if [ "$NOCHAIN" = "1" ] || [ "$STAGE" -ge 5 ]; then
  echo "[merge-sb] stage $STAGE merged; no next-stage training submitted (NOCHAIN=$NOCHAIN stage=$STAGE)."
  exit 0
fi

# Chain the next curriculum stage on the training node with the next rank, warm-starting from the
# merged plain-base checkpoint just produced. Evaluation is a separate step (see eval/ + README).
STEPS_NEXT=$(python3 -c "import math;print(math.ceil(2238036/(8*$BS)))" 2>/dev/null || echo 34969)
NEXT=$((STAGE+1)); NRANK=${RANKS[$((NEXT-1))]}
PARG=""; [ -n "$GPU_PARTITION" ] && PARG="-p $GPU_PARTITION"
cd "$SCR"
# RUN MUST be set explicitly so the next stage does not inherit the previous stage's finished run dir.
sbatch $PARG -w "$TRAIN_NODE" -J "lp_curric_s${NEXT}" \
  --export="ALL,STAGE=$NEXT,RUN=lp_curric_s${NEXT},LRANK=$NRANK,CKPT=$DST,STEPS=$STEPS_NEXT,BS=$BS,DS_ROOT=$DS_ROOT,LR_SCALE=$LR_SCALE,GPU_PARTITION=$GPU_PARTITION,RETRY=0" \
  "$SCR/train.sbatch" \
  && echo "[merge-sb] submitted train_s${NEXT} (rank=$NRANK) on $TRAIN_NODE" \
  || echo "[merge-sb] next-train submit FAILED"
