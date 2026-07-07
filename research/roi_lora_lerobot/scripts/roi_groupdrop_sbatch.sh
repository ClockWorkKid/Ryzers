#!/usr/bin/env bash
#SBATCH --job-name=gd_run
#SBATCH --time=02:00:00
#SBATCH --output=gd_%x_%j.log
# Partial-node SLURM wrapper for the group-drop experiment. Requests GPUs via
# --gres on submit (e.g. sbatch --gres=gpu:1 for the smoke, --gres=gpu:3 for the
# sweep) so it schedules on whatever MI300X node has free GPUs (no full node needed).
# MODE=smoke -> 4-step correctness smoke on 1 GPU.
# MODE=sweep -> 12k-step keep 0.25/0.50/0.75 sweep, one GPU each.
set -u
MODE="${MODE:-smoke}"
# OV = overlay dir (bind-mounted .py), SCR = dir holding the group-drop scripts.
# Env-overridable so this runs on any cluster (set OV / SCR / WORK as needed).
OV="${OV:-$HOME/roi_lerobot_overlay}"
SCR="${SCR:-$HOME}"
export OV
echo "[gd_sbatch] host=$(hostname) job=$SLURM_JOB_ID mode=$MODE OV=$OV SCR=$SCR"
echo "[gd_sbatch] ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-} SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-} GPU_DEVICE_ORDINAL=${GPU_DEVICE_ORDINAL:-}"
# strip Windows CRLF from the scp'd overlay + scripts
cd "$OV" || exit 1
for f in roi_prune.py modeling_molmoact2.py configuration_molmoact2.py hf_modeling_molmoact2.py; do
  tr -d '\r' < "$f" > "$f.tmp" && mv "$f.tmp" "$f"
done
for s in roi_groupdrop_train.sh roi_groupdrop_smoke.sh roi_groupdrop_sweep.sh; do
  [ -f "$SCR/$s" ] && sed -i 's/\r$//' "$SCR/$s"
done
export TRAIN_SH="${TRAIN_SH:-$SCR/roi_groupdrop_train.sh}"

# On multi-node clusters the image may not be pre-baked into every node's local
# docker. If missing, prefer BUILDING it from the on-disk context (base pull over
# internet + pip, ~10min) -- much faster than docker-load of a ~32GB tar off shared
# network storage (~1-2MB/s). Falls back to IMG_TAR load if no build context.
IMG="${IMG:-molmoact2-lerobot-train:rocm942}"
if ! docker image inspect "$IMG" >/dev/null 2>&1; then
  if [ -n "${BUILDCTX:-}" ] && [ -f "$BUILDCTX/Dockerfile.instinct-gfx942" ]; then
    echo "[gd_sbatch] image $IMG absent on $(hostname); BUILD from $BUILDCTX"
    ( cd "$BUILDCTX" && tr -d '\r' < Dockerfile.instinct-gfx942 > Dockerfile.lf && mv Dockerfile.lf Dockerfile.instinct-gfx942
      find roi_overlay -name '*.py' -exec sed -i 's/\r$//' {} + 2>/dev/null
      docker build -t "$IMG" -f Dockerfile.instinct-gfx942 . )
  elif [ -n "${IMG_TAR:-}" ]; then
    echo "[gd_sbatch] image $IMG absent on $(hostname); docker load < $IMG_TAR"
    docker load -i "$IMG_TAR"
  fi
fi

GPULIST="${ROCR_VISIBLE_DEVICES:-${SLURM_JOB_GPUS:-${GPU_DEVICE_ORDINAL:-0}}}"
GPU0="$(echo "$GPULIST" | cut -d, -f1)"

if [ "$MODE" = "sweep" ]; then
  export ROCR_VISIBLE_DEVICES="$GPULIST"
  bash "$SCR/roi_groupdrop_sweep.sh"; RC=$?
elif [ "$MODE" = "train" ]; then
  # single keep-fraction full run on one GPU (for thin/partial-capacity clusters)
  ROI_KEEP="${ROI_KEEP:-0.25}" GPU="$GPU0" STEPS="${STEPS:-12000}" bash "$TRAIN_SH"; RC=$?
else
  GPU="$GPU0" bash "$SCR/roi_groupdrop_smoke.sh"; RC=$?
fi
# Propagate the child's exit code so a failed run marks the SLURM job FAILED.
echo "[gd_sbatch] DONE mode=$MODE rc=$RC"
exit "$RC"
