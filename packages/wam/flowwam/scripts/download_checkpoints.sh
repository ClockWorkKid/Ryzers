#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Fetch the weights FlowWAM's build_pipeline expects into the mounted model dir, in the
# DiffSynth local_model_path layout ($FLOWWAM_MODEL_DIR/<model_id>/<file>). Nothing is
# re-hosted (rules 8/9); everything comes from the original HF repos on demand.
#
#   ryzers run /ryzers/scripts/download_checkpoints.sh [base|stage1|robotwin|embodiments|all]
#
# Components (SeedVR2 refiner deliberately EXCLUDED -- deferred, apex is CUDA-only):
#   base        Wan2.2-TI2V-5B (UMT5-XXL enc + DiT + VAE) + Wan2.1-T2V-1.3B tokenizer  (~12 GB)
#   stage1      FlowWAM world-MODEL checkpoint (WorldArena open-loop video eval)
#   robotwin    FlowWAM world-ACTION checkpoint + action-norm stats (RoboTwin closed-loop policy)
#   embodiments RoboTwin2.0 embodiment URDFs (SAPIEN robot-only renderer, ~220 MB)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HF_COMMON:-$HERE/_hf_common.sh}"

WHICH="${1:-all}"
MDIR="${FLOWWAM_MODEL_DIR:-/models/flowwam}"
mkdir -p "$MDIR"

dl_base() {
  # DiffSynth's WanVideoPipeline.from_pretrained redirects the shared UMT5-XXL text encoder
  # (models_t5_umt5-xxl-enc-bf16.pth) to the Wan2.1-T2V-1.3B repo (redirect_common_files=True),
  # so it must live under Wan2.1-T2V-1.3B (it is NOT in the Wan2.2-TI2V-5B repo). The Wan2.2 DiT
  # (diffusion_pytorch_model*) + VAE (Wan2.2_VAE.pth) stay under Wan2.2-TI2V-5B.
  echo "===== Wan2.2-TI2V-5B (DiT + VAE + configs) ====="
  hf_prefetch Wan-AI/Wan2.2-TI2V-5B \
    --include "diffusion_pytorch_model*.safetensors" "*.index.json" \
              "Wan2.2_VAE.pth" "config.json" "configuration.json" \
    --local-dir "$MDIR/Wan-AI/Wan2.2-TI2V-5B"
  echo "===== Wan2.1-T2V-1.3B (UMT5-XXL text encoder + google tokenizer) ====="
  hf_prefetch Wan-AI/Wan2.1-T2V-1.3B \
    --include "google/*" "models_t5_umt5-xxl-enc-bf16.pth" \
    --local-dir "$MDIR/Wan-AI/Wan2.1-T2V-1.3B"
  # Layout note: the OPEN-LOOP world-model build_pipeline redirects the shared T5 to Wan2.1
  # (redirect_common_files=True), but the CLOSED-LOOP flow-action server's pipeline_loader loads
  # models_t5 with model_id=Wan-AI/Wan2.2-TI2V-5B + explicit local_model_path (no redirect), i.e.
  # it expects the file under Wan2.2-TI2V-5B/. Expose the same weight under both layouts so either
  # eval mode works from one cache (symlink; no re-download, rules 8/9).
  local t5="models_t5_umt5-xxl-enc-bf16.pth"
  if [ -f "$MDIR/Wan-AI/Wan2.1-T2V-1.3B/$t5" ] && [ ! -e "$MDIR/Wan-AI/Wan2.2-TI2V-5B/$t5" ]; then
    ln -sfn "../Wan2.1-T2V-1.3B/$t5" "$MDIR/Wan-AI/Wan2.2-TI2V-5B/$t5"
    echo "    linked $t5 into Wan2.2-TI2V-5B/ (closed-loop server layout)"
  fi
}

dl_stage1() {
  echo "===== FlowWAM world-MODEL checkpoint (WorldArena open-loop video eval) ====="
  hf_prefetch YixiangChen/FlowWAM flowwam_worldarena_stage1.safetensors \
    --local-dir "$MDIR/stage_1"
}

dl_robotwin() {
  # Closed-loop ACTION policy: the dual-stream DiT + IDM action expert weights and the action
  # normalization stats the flow_action_server expects (README "Download the FlowWAM checkpoint").
  echo "===== FlowWAM world-ACTION checkpoint (RoboTwin closed-loop policy) ====="
  hf_prefetch YixiangChen/FlowWAM \
    flowwam_robotwin.safetensors flowwam_robotwin_action_norm_stats.npz \
    --local-dir "$MDIR/robotwin"
}

dl_embodiments() {
  echo "===== RoboTwin2.0 embodiments (SAPIEN URDFs) ====="
  hf_prefetch TianxingChen/RoboTwin2.0 embodiments.zip \
    --repo-type dataset --local-dir "$MDIR/embodiments_dl"
  if [ -f "$MDIR/embodiments_dl/embodiments.zip" ]; then
    # Use python's zipfile so we don't depend on the `unzip` apt package being present.
    if command -v unzip >/dev/null 2>&1; then
      unzip -q -o "$MDIR/embodiments_dl/embodiments.zip" -d "$MDIR/embodiments"
    else
      python3 -m zipfile -e "$MDIR/embodiments_dl/embodiments.zip" "$MDIR/embodiments/"
    fi
    echo "    extracted -> $MDIR/embodiments"
  fi
}

case "$WHICH" in
  base)        dl_base ;;
  stage1)      dl_stage1 ;;
  robotwin)    dl_robotwin ;;
  embodiments) dl_embodiments ;;
  all)         dl_base; dl_stage1; dl_robotwin; dl_embodiments ;;
  *) echo "usage: download_checkpoints.sh [base|stage1|robotwin|embodiments|all]" >&2; exit 2 ;;
esac

echo "PASS: FlowWAM weights ($WHICH) cached under $MDIR"
find "$MDIR" -maxdepth 3 -type f | head -n 40
