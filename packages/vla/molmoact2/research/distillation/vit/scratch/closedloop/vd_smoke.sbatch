#!/usr/bin/env bash
#SBATCH --job-name=vd_smoke
#SBATCH -p mi210
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 00:40:00
#SBATCH --output=/shared_nobackup/%u/molmoact2/vd_smoke_%j.log
# Validate the distillation runtime on mi210 via apptainer + eval SIF:
#  1) install torchdistill into a shared --target dir (bound on PYTHONPATH)
#  2) build all 3 student variants + forward-shape check
#  3) tiny end-to-end distillation (teacher load + data + loss + ckpt)
set -u
U=$USER
WORK=/shared_nobackup/$U/molmoact2
SIF=$WORK/images/molmoact2-lerobot-eval.sif
CACHE=$WORK/hf_cache
CODE=$WORK/vit_distill
TD=$WORK/pylibs
mkdir -p "$TD" "$CODE/resource/ckpt/vit_distill" "$CODE/logs"
echo "[vd_smoke] host=$(hostname) job=${SLURM_JOB_ID:-none} rocr=${ROCR_VISIBLE_DEVICES:-} t=$(date -Is)"

echo "=== 1) ensure torchdistill in $TD ==="
apptainer exec --cleanenv --home "$CACHE":/cache --env PYTHONPATH=/td --bind "$TD":/td \
  "$SIF" bash -lc 'python -c "import torchdistill" 2>/dev/null && { echo already; exit 0; }; \
    pip install --target=/td --no-deps torchdistill && \
    python -c "import yaml" 2>/dev/null || pip install --target=/td --no-deps pyyaml; \
    python -c "import scipy" 2>/dev/null || pip install --target=/td --no-deps scipy' 2>&1 | tail -15

echo "=== 2) student variant build + forward-shape check ==="
apptainer exec --rocm --cleanenv --home "$CACHE":/cache \
  --env PYTHONPATH=/work:/td --bind "$CODE":/work --bind "$TD":/td --pwd /work \
  "$SIF" bash -lc 'python - <<"PY"
import torch
from distill.config import StudentConfig
from distill.student import build_seam_student
cases = {
  "hybrid":  dict(variant="hybrid", dim=256, num_conv_blocks=4, num_attn_blocks=3, attn_heads=8),
  "siglip_nano": dict(variant="tinyvit", dim=240, num_attn_blocks=4, attn_heads=8, attn_mlp_ratio=2.0),
  "cnn_fpga": dict(variant="cnn", dim=384, num_conv_blocks=16, conv_kernel=3),
}
x = torch.randn(2, 1, 729, 588)
for name, kw in cases.items():
    m = build_seam_student(**kw)
    y = m(x)
    p = m.num_params()
    status = "OK" if tuple(y.shape) == (2, 1, 729, 2304) else "BAD"
    print(f"{name:12s} params={p/1e6:.3f}M out={tuple(y.shape)} {status}")
PY' 2>&1 | tail -20

echo "=== 3) tiny end-to-end distillation (distill_smoke.yaml) ==="
apptainer exec --rocm --cleanenv --home "$CACHE":/cache \
  --env HF_HOME=/cache --env XDG_CACHE_HOME=/cache \
  --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
  --env TORCH_BLAS_PREFER_HIPBLASLT=0 --env PYTORCH_ALLOC_CONF=expandable_segments:True \
  --env PYTHONPATH=/work:/td --bind "$CODE":/work --bind "$TD":/td --pwd /work \
  "$SIF" torchrun --standalone --nproc_per_node=1 train.py \
    --config configs/distill_smoke.yaml --world_size 1 --run_log logs/smoke.log 2>&1 | tail -30
echo "[vd_smoke] DONE rc=$? t=$(date -Is)"
