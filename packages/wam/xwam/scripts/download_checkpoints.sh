#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Fetch the released X-WAM checkpoints (sharinka0715/X-WAM-checkpoints) and the Wan2.2-TI2V-5B
# base weights (Wan-AI/Wan2.2-TI2V-5B: UMT5-XXL text encoder + Wan2.2 VAE + DiT base) into the
# mounted model volume (rule 8: never baked into the image). The base is required by
# XWAMModel.from_pretrained / the T5 + VAE loaders.
#   ryzers run /ryzers/scripts/download_checkpoints.sh [all|pretrained|robotwin|robocasa|base]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HF_COMMON:-$HERE/_hf_common.sh}"

WHICH="${1:-all}"
CKPT_ROOT="${CKPT_ROOT:-/models/xwam/checkpoints}"
WAN_CKPT_DIR="${WAN_CKPT_DIR:-/models/xwam/wan22_5b}"
mkdir -p "$CKPT_ROOT" "$WAN_CKPT_DIR"

fetch_base() {
  # UMT5-XXL enc + Wan2.2 VAE + DiT base. Names match configs/model/wan22_5b_sft.yaml
  # (t5_checkpoint=models_t5_umt5-xxl-enc-bf16.pth, vae_checkpoint=Wan2.2_VAE.pth).
  hf_prefetch Wan-AI/Wan2.2-TI2V-5B --local-dir "$WAN_CKPT_DIR"
}

fetch_ckpt() {  # $1 = subdir under the HF repo (robotwin_sft|robocasa_sft|pretrained)
  hf_prefetch sharinka0715/X-WAM-checkpoints --include "$1/*" --local-dir "$CKPT_ROOT"
}

case "$WHICH" in
  base)       fetch_base ;;
  pretrained) fetch_ckpt pretrained ;;
  robotwin)   fetch_base; fetch_ckpt robotwin_sft ;;
  robocasa)   fetch_base; fetch_ckpt robocasa_sft ;;
  all)        fetch_base; hf_prefetch sharinka0715/X-WAM-checkpoints --local-dir "$CKPT_ROOT" ;;
  *) echo "usage: download_checkpoints.sh [all|pretrained|robotwin|robocasa|base]" >&2; exit 2 ;;
esac

echo "PASS: X-WAM checkpoints ($WHICH) cached (CKPT_ROOT=$CKPT_ROOT, WAN_CKPT_DIR=$WAN_CKPT_DIR)"
ls -la "$CKPT_ROOT" || true
