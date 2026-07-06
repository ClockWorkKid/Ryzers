#!/usr/bin/env bash
# Runs INSIDE one srun allocation on an MI300X node (8 GPUs). Strips CRLF from the
# scp'd overlay, then launches the three group-drop keep-fraction training runs
# concurrently (one GPU each) and waits. Mirrors roi_gatedistill_sweep.sh but with
# GROUP-DROP on + the longer (12000-step) schedule.
#
# Launch (from the login node), e.g.:
#   srun --nodelist=<MI300X-node> --gres=gpu:3 --time=02:00:00 --pty \
#        bash ~/roi_groupdrop_sweep.sh
# or chain 2h segments with sbatch --wrap of this script (each segment resumes).
set -u
OV=$HOME/roi_lerobot_overlay
OUT=/shared_nobackup/user/molmoact2/outputs
cd "$OV" || exit 1
for f in roi_prune.py modeling_molmoact2.py configuration_molmoact2.py hf_modeling_molmoact2.py; do
  tr -d '\r' < "$f" > "$f.tmp" && mv "$f.tmp" "$f"
done
sed -i 's/\r$//' "$HOME/roi_groupdrop_train.sh"
# Pin each run to a SLURM-assigned physical GPU (works for partial-node allocations:
# docker --device=/dev/dri exposes all GPUs, so HIP_VISIBLE_DEVICES must be the
# physical index SLURM granted, not a 0-based loop counter).
GPUS_ASSIGNED="${ROCR_VISIBLE_DEVICES:-${SLURM_JOB_GPUS:-${GPU_DEVICE_ORDINAL:-0,1,2}}}"
IFS=',' read -ra GARR <<< "$GPUS_ASSIGNED"
echo "[gdsweep] host=$(hostname) assigned GPUs=[$GPUS_ASSIGNED] overlay stripped; launching group-drop runs"

pids=()
i=0
for KEEP in 0.25 0.5 0.75; do
  GPU="${GARR[$i]:-$i}"
  PORT=$((29760 + i))
  ROI_KEEP="$KEEP" GPU="$GPU" PORT="$PORT" STEPS="${STEPS:-12000}" bash "$HOME/roi_groupdrop_train.sh" \
    > "$OUT/sweep_groupdrop_keep${KEEP}.wrap.log" 2>&1 &
  pids+=("$!")
  echo "[gdsweep] launched keep=$KEEP gpu=$GPU port=$PORT pid=$!"
  i=$((i + 1))
  sleep 25
done

echo "[gdsweep] waiting on pids: ${pids[*]}"
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "[gdsweep] SWEEP_ALL_DONE rc=$rc"
