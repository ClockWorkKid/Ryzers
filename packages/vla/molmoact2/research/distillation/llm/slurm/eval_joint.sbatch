#!/usr/bin/env bash
#SBATCH --job-name=llmd_eval
#SBATCH -p gpu                       # <- set to your Slurm GPU partition (runs used a single MI210)
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -c 16
#SBATCH --mem=120G
#SBATCH -t 01:00:00
#SBATCH --output=logs/llmd_eval_%x_%j.log
# Closed-loop LIBERO eval of the distilled student (or teacher baseline). Overlay-bound
# teacher modeling (train/deploy consistency). A full gate = 4 suites x 10 tasks x 3 ep
# (loop this script over suites/tasks, or set TASK_IDS="0 1 2 3 4 5 6 7 8 9").
# Env: STUDENT_CKPT, SUITE, TASK_IDS, N_EP, SEED, OUTDIR, TEACHER_BASELINE(0/1).
set -u
U=$USER
WORK="${WORK:-$HOME/molmoact2}"
SIF="${SIF:-$WORK/images/molmoact2-lerobot-eval.sif}"
CODE="${CODE:-$WORK/llm_distill}"
CACHE="${CACHE:-$WORK/hf_cache}"
OUTPUTS="${OUTPUTS:-$WORK/outputs}"
OV="${OV:-$WORK/roi_overlay}"
MP=/opt/lerobot/src/lerobot/policies/molmoact2
GPU=$(echo "${ROCR_VISIBLE_DEVICES:-0}" | awk -F, '{print $1}')
SUITE="${SUITE:-libero_spatial}"; TASK_IDS="${TASK_IDS:-0}"; N_EP="${N_EP:-5}"
SEED="${SEED:-1000}"; OUTDIR="${OUTDIR:-/outputs/llm_distill/eval/tmp}"
STUDENT_CKPT="${STUDENT_CKPT:-/outputs/llm_distill/runs/qwen06w_fn/latest.pt}"
TB=""; [ "${TEACHER_BASELINE:-0}" = "1" ] && TB="--teacher-baseline"
echo "[eval] host=$(hostname) job=${SLURM_JOB_ID:-none} suite=$SUITE tasks=$TASK_IDS n_ep=$N_EP tb=${TEACHER_BASELINE:-0} gpu=$GPU t=$(date -Is)"
apptainer exec --rocm --cleanenv --home "$CACHE":/cache \
  --env USER="$U" --env LOGNAME="$U" \
  --env HF_HOME=/cache --env XDG_CACHE_HOME=/cache \
  --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
  --env HF_HUB_DOWNLOAD_TIMEOUT=120 --env TORCH_BLAS_PREFER_HIPBLASLT=0 \
  --env PYTORCH_ALLOC_CONF=expandable_segments:True \
  --env MUJOCO_GL="${MUJOCO_GL:-osmesa}" --env PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}" \
  --env OMP_NUM_THREADS=1 --env MKL_NUM_THREADS=1 \
  --env HIP_VISIBLE_DEVICES=$GPU --env PYTHONPATH=/work \
  --bind "$CODE":/work --bind "$OUTPUTS":/outputs --pwd /work \
  --bind "$OV/roi_prune.py":"$MP/roi_prune.py":ro \
  --bind "$OV/modeling_molmoact2.py":"$MP/modeling_molmoact2.py":ro \
  --bind "$OV/configuration_molmoact2.py":"$MP/configuration_molmoact2.py":ro \
  --bind "$OV/hf_modeling_molmoact2.py":"$MP/molmoact2_hf_model/modeling_molmoact2.py":ro \
  "$SIF" bash -lc "if [ ! -f \$HOME/.libero/config.yaml ]; then printf 'N\n' | python3 -c 'import libero.libero' >/dev/null 2>&1 || true; fi; \
    python /work/eval_joint.py $TB --student-ckpt '$STUDENT_CKPT' --suite '$SUITE' \
      --task-ids '$TASK_IDS' --n-episodes $N_EP --seed $SEED --output-dir '$OUTDIR' --device cuda"
RC=$?
echo "[eval] DONE rc=$RC t=$(date -Is)"
exit $RC
