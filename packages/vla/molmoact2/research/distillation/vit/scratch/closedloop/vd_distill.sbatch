#!/usr/bin/env bash
#SBATCH --job-name=vd_distill
#SBATCH -p mi210
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH -c 48
#SBATCH --mem=256G
#SBATCH -t 04:00:00
#SBATCH --output=/shared_nobackup/%u/molmoact2/vd_distill_%x_%j.log
# 10-epoch vision-encoder distillation on one mi210 node (4-GPU DDP) via apptainer
# + eval SIF + shared torchdistill target. mi210 QOS caps jobs at 4h; train.py saves
# the best (highest val-cosine) student every improving epoch, so a 4h wall-stop
# still yields a near-converged checkpoint (orig curve hit ~0.88 by epoch 6-7).
# Parametrized by CFG + NAME.
#   sbatch --job-name=d_siglip --export=ALL,CFG=configs/distill_siglip_nano.yaml,NAME=siglip_nano vd_distill.sbatch
set -u
U=$USER
WORK=/shared_nobackup/$U/molmoact2
SIF=$WORK/images/molmoact2-lerobot-eval.sif
CACHE=$WORK/hf_cache
CODE=$WORK/vit_distill
TD=$WORK/pylibs
CFG="${CFG:-configs/distill_siglip_nano.yaml}"
NAME="${NAME:-siglip_nano}"
NG=$(echo "${ROCR_VISIBLE_DEVICES:-0,1,2,3}" | awk -F, '{print NF}')
mkdir -p "$TD" "$CODE/resource/ckpt/vit_distill" "$CODE/logs"
echo "[vd_distill] host=$(hostname) job=${SLURM_JOB_ID:-none} name=$NAME cfg=$CFG ngpu=$NG t=$(date -Is)"

# ensure torchdistill available in the shared target (idempotent, online on mi210)
apptainer exec --cleanenv --home "$CACHE":/cache --env PYTHONPATH=/td --bind "$TD":/td \
  "$SIF" bash -lc 'python -c "import torchdistill" 2>/dev/null || { \
     pip install --target=/td --no-deps torchdistill; \
     python -c "import yaml" 2>/dev/null || pip install --target=/td --no-deps pyyaml; \
     python -c "import scipy" 2>/dev/null || pip install --target=/td --no-deps scipy; }' 2>&1 | tail -6

# full distillation
apptainer exec --rocm --cleanenv --home "$CACHE":/cache \
  --env HF_HOME=/cache --env XDG_CACHE_HOME=/cache \
  --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
  --env HF_HUB_DOWNLOAD_TIMEOUT=120 --env TORCH_BLAS_PREFER_HIPBLASLT=0 \
  --env PYTORCH_ALLOC_CONF=expandable_segments:True \
  --env PYTHONPATH=/work:/td --bind "$CODE":/work --bind "$TD":/td --pwd /work \
  "$SIF" torchrun --standalone --nproc_per_node=$NG train.py \
    --config "$CFG" --world_size $NG --run_log "logs/${NAME}.log"
RC=$?
echo "[vd_distill] DONE name=$NAME rc=$RC t=$(date -Is)"
exit $RC
