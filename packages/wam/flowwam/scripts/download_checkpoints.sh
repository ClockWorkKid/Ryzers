#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Fetch the weights FlowWAM's build_pipeline expects into the mounted model dir, in the
# DiffSynth local_model_path layout ($FLOWWAM_MODEL_DIR/<model_id>/<file>). Nothing is
# re-hosted (rules 8/9); everything comes from the original HF repos on demand.
#
#   ryzers run /ryzers/scripts/download_checkpoints.sh [base|stage1|embodiments|all]
#
# Components (SeedVR2 refiner deliberately EXCLUDED -- deferred, apex is CUDA-only):
#   base        Wan2.2-TI2V-5B (UMT5-XXL enc + DiT + VAE) + Wan2.1-T2V-1.3B tokenizer  (~12 GB)
#   stage1      FlowWAM stage-1 world-model checkpoint (YixiangChen/FlowWAM)
#   embodiments RoboTwin2.0 embodiment URDFs (SAPIEN robot-only renderer, ~220 MB)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HF_COMMON:-$HERE/_hf_common.sh}"

WHICH="${1:-all}"
MDIR="${FLOWWAM_MODEL_DIR:-/models/flowwam}"
mkdir -p "$MDIR"

dl_base() {
  echo "===== Wan2.2-TI2V-5B base (enc + DiT + VAE) ====="
  hf_prefetch Wan-AI/Wan2.2-TI2V-5B \
    --include "models_t5_umt5-xxl-enc-bf16.pth" "diffusion_pytorch_model*.safetensors" \
              "Wan2.2_VAE.pth" "config.json" "*.json" \
    --local-dir "$MDIR/Wan-AI/Wan2.2-TI2V-5B"
  echo "===== Wan2.1-T2V-1.3B tokenizer (google/*) ====="
  hf_prefetch Wan-AI/Wan2.1-T2V-1.3B \
    --include "google/*" \
    --local-dir "$MDIR/Wan-AI/Wan2.1-T2V-1.3B"
}

dl_stage1() {
  echo "===== FlowWAM stage-1 checkpoint ====="
  hf_prefetch YixiangChen/FlowWAM flowwam_worldarena_stage1.safetensors \
    --local-dir "$MDIR/stage_1"
}

dl_embodiments() {
  echo "===== RoboTwin2.0 embodiments (SAPIEN URDFs) ====="
  hf_prefetch TianxingChen/RoboTwin2.0 embodiments.zip \
    --repo-type dataset --local-dir "$MDIR/embodiments_dl"
  if [ -f "$MDIR/embodiments_dl/embodiments.zip" ]; then
    ( cd "$MDIR/embodiments_dl" && unzip -q -o embodiments.zip -d "$MDIR/embodiments" )
    echo "    extracted -> $MDIR/embodiments"
  fi
}

case "$WHICH" in
  base)        dl_base ;;
  stage1)      dl_stage1 ;;
  embodiments) dl_embodiments ;;
  all)         dl_base; dl_stage1; dl_embodiments ;;
  *) echo "usage: download_checkpoints.sh [base|stage1|embodiments|all]" >&2; exit 2 ;;
esac

echo "PASS: FlowWAM weights ($WHICH) cached under $MDIR"
find "$MDIR" -maxdepth 3 -type f | head -n 40
