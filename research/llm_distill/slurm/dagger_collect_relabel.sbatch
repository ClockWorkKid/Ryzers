#!/usr/bin/env bash
#SBATCH --job-name=dagger_cr
#SBATCH -p gpu                       # <- set to your Slurm GPU partition (runs used a single MI210)
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -c 16
#SBATCH --mem=160G
#SBATCH -t 04:00:00
#SBATCH --output=logs/dagger_cr_%x_%j.log
# Stage 3, steps 1-2: COLLECT student-visited states closed-loop, then RELABEL them with
# the teacher's current-step normalized action. Set CKPT to the base policy to roll out.
# Full run used: SUITE="libero_spatial,libero_object,libero_goal,libero_10" TASK_IDS=0..9 N_EP=3.
# Env: CKPT, SUITE, TASK_IDS, N_EP, MAXSTEPS, LIMIT, OUT.
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
CKPT="${CKPT:-/outputs/llm_distill/runs/qwen06w_expd/step_12000.pt}"
SUITE="${SUITE:-libero_spatial,libero_object,libero_goal,libero_10}"
TASK_IDS="${TASK_IDS:-0,1,2,3,4,5,6,7,8,9}"; N_EP="${N_EP:-3}"; MAXSTEPS="${MAXSTEPS:-520}"
LIMIT="${LIMIT:-0}"; OUT="${OUT:-/outputs/llm_distill/dagger/expd}"
echo "[dagger_cr] host=$(hostname) job=${SLURM_JOB_ID:-none} ckpt=$CKPT out=$OUT gpu=$GPU t=$(date -Is)"
apptainer exec --rocm --cleanenv --home "$CACHE":/cache \
  --env USER="$U" --env LOGNAME="$U" \
  --env HF_HOME=/cache --env XDG_CACHE_HOME=/cache --env MPLCONFIGDIR=/cache/mpl \
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
  "$SIF" bash -lc "set -e; \
    echo '=== STEP 1: collect on-policy states ==='; \
    python /work/collect_dagger.py --student-ckpt '$CKPT' --suites '$SUITE' \
      --task-ids '$TASK_IDS' --n-episodes $N_EP --max-steps $MAXSTEPS \
      --out '$OUT/collect' --device cuda; \
    echo '=== STEP 2: validate + relabel with teacher ==='; \
    python /work/relabel_teacher.py --validate --collect-dir '$OUT/collect' \
      --out '$OUT/relabel' ${LIMIT:+--limit $LIMIT} --device cuda"
RC=$?
echo "[dagger_cr] DONE rc=$RC t=$(date -Is)"
exit $RC
