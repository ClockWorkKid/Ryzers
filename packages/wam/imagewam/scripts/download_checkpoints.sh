#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Fetch the FLUX.2 ImageWAM stack into the mounted model volume:
#   1. ImageWAM release checkpoint (public): model.pt + dataset_stats.json + train_config.yaml
#   2. FLUX.2 klein-base DiT weights            (GATED: needs HF_TOKEN with granted access)
#   3. FLUX.2-dev autoencoder (ae.safetensors)  (GATED: needs HF_TOKEN with granted access)
# The Qwen3 text encoder (Qwen/Qwen3-4B) is fetched by the upstream loader on first model run.
#   ryzers run /ryzers/scripts/download_checkpoints.sh [libero|robotwin] [4b|9b]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HF_COMMON:-$HERE/_hf_common.sh}"

SUITE="${1:-libero}"                 # libero | robotwin
VARIANT="${2:-${FLUX2_VARIANT:-4b}}" # 4b | 9b
MODELS="${MODELS_DIR:-/models}"
FLUX2_DIR="${FLUX2_DIR:-$MODELS/flux2}"
REL_DIR="${IMAGEWAM_RELEASE_DIR:-$MODELS/imagewam_release}"

# ---- 1) ImageWAM released checkpoint (public) --------------------------------------------
case "${SUITE}_${VARIANT}" in
  libero_4b)   REPO="yuyangalin/ImageWAM-FLUX.2-4B-LIBERO";   DEST="$REL_DIR/libero/flux2_klein_4b" ;;
  robotwin_4b) REPO="yuyangalin/ImageWAM-FLUX.2-4B-RoboTwin"; DEST="$REL_DIR/robotwin/flux2_klein_4b" ;;
  libero_9b)   REPO="yuyangalin/ImageWAM-FLUX.2-9B-LIBERO";   DEST="$REL_DIR/libero/flux2_klein_9b" ;;
  *) echo "usage: download_checkpoints.sh [libero|robotwin] [4b|9b] (robotwin only has 4b released)" >&2; exit 2 ;;
esac
mkdir -p "$DEST"
hf_prefetch "$REPO" --repo-type model --local-dir "$DEST"
echo "PASS: ImageWAM checkpoint ($SUITE $VARIANT) cached under $DEST"

# ---- 2/3) FLUX.2 base DiT + autoencoder (GATED) ------------------------------------------
if [ -z "${HF_TOKEN:-}" ]; then
  echo "WARN: HF_TOKEN unset. FLUX.2 base + AE are GATED (black-forest-labs) and will be skipped." >&2
  echo "      Set HF_TOKEN (with granted access) to fetch them, then re-run." >&2
  exit 0
fi

if [ "$VARIANT" = "9b" ]; then
  FLUX2_BASE_REPO="black-forest-labs/FLUX.2-klein-base-9B"; FLUX2_BASE_FILE="flux-2-klein-base-9b.safetensors"; FLUX2_BASE_SUB="FLUX.2-klein-base-9B"
else
  FLUX2_BASE_REPO="black-forest-labs/FLUX.2-klein-base-4B"; FLUX2_BASE_FILE="flux-2-klein-base-4b.safetensors"; FLUX2_BASE_SUB="FLUX.2-klein-base-4B"
fi
hf_prefetch "$FLUX2_BASE_REPO" "$FLUX2_BASE_FILE" --local-dir "$FLUX2_DIR/$FLUX2_BASE_SUB"
hf_prefetch black-forest-labs/FLUX.2-dev ae.safetensors --local-dir "$FLUX2_DIR/FLUX.2-dev"
echo "PASS: FLUX.2 $VARIANT base + AE cached under $FLUX2_DIR"
ls -la "$DEST" "$FLUX2_DIR/$FLUX2_BASE_SUB" "$FLUX2_DIR/FLUX.2-dev" 2>/dev/null || true
