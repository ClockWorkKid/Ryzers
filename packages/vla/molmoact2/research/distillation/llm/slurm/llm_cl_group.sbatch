#!/usr/bin/env bash
#SBATCH --job-name=llm_clg
#SBATCH -p gpu            # set to your 4-GPU partition (runs were on MI210 nodes)
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH -c 64
#SBATCH --mem=256G
#SBATCH -t 03:55:00
#SBATCH --output=llm_clg_%x_%j.log
# Closed-loop LIBERO eval for a GROUP of quantized LLM-backbone cells, run
# sequentially on one 4-GPU node (each cell fans its 4 suites across the 4 GPUs).
#   ARMS="fp: qat_w8a8:/outputs/llm_quant/mixedft_w8a8_io8/best.pt ..." NEP=10 \
#     sbatch --export=ALL llm_cl_group.sbatch
# arm spec = tag:blobpath ; an empty blob (just 'fp:') is the full-precision baseline.
#
# Paths below are overridable via env; point WORK at your own layout.
set -u
U=$USER
WORK="${WORK:-$HOME/molmoact2}"
SIF="${SIF:-$WORK/images/molmoact2-lerobot-eval.sif}"
CODE="${CODE:-$WORK/llm_distill}"
CACHE="${CACHE:-$WORK/hf_cache}"
PYLIBS="${PYLIBS:-$WORK/pylibs_quant}"
OV="${OV:-$HOME/roi_lerobot_overlay}"
MP=/opt/lerobot/src/lerobot/policies/molmoact2
ARMS="${ARMS:?set ARMS as tag:blob specs (empty blob = FP baseline)}"
NEP="${NEP:-4}"
echo "[llm_clg] host=$(hostname) arms='$ARMS' nep=$NEP job=${SLURM_JOB_ID:-none} t=$(date -Is)"
apptainer exec --rocm --cleanenv --home "$CACHE":/cache \
  --env USER="$U" --env LOGNAME="$U" --env HF_HOME=/cache --env XDG_CACHE_HOME=/cache \
  --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env HF_HUB_DOWNLOAD_TIMEOUT=120 \
  --env TORCH_BLAS_PREFER_HIPBLASLT=0 --env PYTORCH_ALLOC_CONF=expandable_segments:True \
  --env MUJOCO_GL=osmesa --env PYOPENGL_PLATFORM=osmesa \
  --env OMP_NUM_THREADS=1 --env MKL_NUM_THREADS=1 --env PYTHONPATH=/work:/pylibs \
  --env ARMS="$ARMS" --env NEP="$NEP" \
  --bind "$CODE":/work --bind "$WORK/outputs":/outputs --bind "$PYLIBS":/pylibs --pwd /work \
  --bind "$OV/roi_prune.py":"$MP/roi_prune.py":ro \
  --bind "$OV/modeling_molmoact2.py":"$MP/modeling_molmoact2.py":ro \
  --bind "$OV/configuration_molmoact2.py":"$MP/configuration_molmoact2.py":ro \
  --bind "$OV/hf_modeling_molmoact2.py":"$MP/molmoact2_hf_model/modeling_molmoact2.py":ro \
  "$SIF" bash /work/llm_cl_grouprun.sh
RC=$?
echo "[llm_clg] DONE rc=$RC t=$(date -Is)"
exit $RC
