#!/usr/bin/env bash
#SBATCH --job-name=vd_eval
#SBATCH -p mi210
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH -c 64
#SBATCH --mem=256G
#SBATCH -t 04:00:00
#SBATCH --output=/shared_nobackup/%u/molmoact2/vd_eval_%x_%j.log
# One closed-loop eval run (4 GPUs, one node). Submit twice for the parallel
# teacher-vs-student comparison:
#   sbatch -J vd_teacher --export=ALL,MODE=teacher vd_eval.sbatch
#   sbatch -J vd_student --export=ALL,MODE=student vd_eval.sbatch
set -u
U=$USER
CACHE=/shared_nobackup/$U/molmoact2/hf_cache
OUT=/shared_nobackup/$U/molmoact2/outputs
SIF=/shared_nobackup/$U/molmoact2/images/molmoact2-lerobot-eval.sif
MODE="${MODE:-teacher}"
cp -f "$HOME/vd_eval_sweep.sh" "$OUT/vd_eval_sweep.sh"; sed -i 's/\r$//' "$OUT/vd_eval_sweep.sh"
NG=$(echo "${ROCR_VISIBLE_DEVICES:-0,1,2,3}" | awk -F, '{print NF}')
echo "[vd_eval] host=$(hostname) mode=$MODE job=${SLURM_JOB_ID:-none} rocr=${ROCR_VISIBLE_DEVICES:-} ngpu=$NG t=$(date -Is)"
[ -f "$SIF" ] || { echo "[vd_eval] SIF missing: $SIF"; exit 2; }
[ -f "$OUT/base_teacher/checkpoints/000000/pretrained_model/config.json" ] || { echo "[vd_eval] teacher ckpt missing"; exit 3; }
apptainer exec --rocm --cleanenv --home "$CACHE":/cache \
  --env USER="$U" --env LOGNAME="$U" \
  --env HF_HOME=/cache --env XDG_CACHE_HOME=/cache \
  --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
  --env HF_HUB_DOWNLOAD_TIMEOUT=120 --env TORCH_BLAS_PREFER_HIPBLASLT=0 \
  --env PYTORCH_ALLOC_CONF=expandable_segments:True \
  --env MUJOCO_GL=osmesa --env PYOPENGL_PLATFORM=osmesa \
  --env MODE="$MODE" --env N_EP="${N_EP:-5}" --env SEED="${SEED:-1000}" \
  --env NGPU="$NG" --env WPG="${WPG:-1}" \
  --env RESDIR="${RESDIR:-/outputs/vd_eval_$MODE}" \
  --env CKPT="${CKPT:-/outputs/base_teacher/checkpoints/000000/pretrained_model}" \
  --env SUITES="${SUITES:-libero_spatial libero_object libero_goal libero_10}" \
  --env STD_TASKS="${STD_TASKS:-0 1 2 3 4}" \
  --env VIT_STUDENT_CKPT="${VIT_STUDENT_CKPT:-/outputs/hybrid_full.pt}" \
  --env VIT_STUDENT_MODULE="${VIT_STUDENT_MODULE:-/outputs/vit_student.py}" \
  --bind "$OUT":/outputs \
  "$SIF" bash -lc "bash /outputs/vd_eval_sweep.sh"
echo "[vd_eval] DONE rc=$? t=$(date -Is)"
