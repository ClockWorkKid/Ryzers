#!/usr/bin/env bash
#SBATCH --job-name=gd_run
#SBATCH --time=02:00:00
#SBATCH --output=/shared_nobackup/user/molmoact2/outputs/gd_%x_%j.log
# Partial-node SLURM wrapper for the group-drop experiment. Requests GPUs via
# --gres on submit (e.g. sbatch --gres=gpu:1 for the smoke, --gres=gpu:3 for the
# sweep) so it schedules on whatever MI300X node has free GPUs (no full node needed).
# MODE=smoke -> 4-step correctness smoke on 1 GPU.
# MODE=sweep -> 12k-step keep 0.25/0.50/0.75 sweep, one GPU each.
set -u
MODE="${MODE:-smoke}"
echo "[gd_sbatch] host=$(hostname) job=$SLURM_JOB_ID mode=$MODE"
echo "[gd_sbatch] ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-} SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-} GPU_DEVICE_ORDINAL=${GPU_DEVICE_ORDINAL:-}"
# strip Windows CRLF from the scp'd overlay + scripts
cd "$HOME/roi_lerobot_overlay" || exit 1
for f in roi_prune.py modeling_molmoact2.py configuration_molmoact2.py hf_modeling_molmoact2.py; do
  tr -d '\r' < "$f" > "$f.tmp" && mv "$f.tmp" "$f"
done
for s in roi_groupdrop_train.sh roi_groupdrop_smoke.sh roi_groupdrop_sweep.sh; do
  [ -f "$HOME/$s" ] && sed -i 's/\r$//' "$HOME/$s"
done

GPULIST="${ROCR_VISIBLE_DEVICES:-${SLURM_JOB_GPUS:-${GPU_DEVICE_ORDINAL:-0}}}"
GPU0="$(echo "$GPULIST" | cut -d, -f1)"

if [ "$MODE" = "sweep" ]; then
  export ROCR_VISIBLE_DEVICES="$GPULIST"
  bash "$HOME/roi_groupdrop_sweep.sh"
else
  GPU="$GPU0" bash "$HOME/roi_groupdrop_smoke.sh"
fi
echo "[gd_sbatch] DONE mode=$MODE rc=$?"
