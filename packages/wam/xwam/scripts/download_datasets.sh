#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Fetch a small open-loop subset of the released X-WAM datasets into the mounted model volume
# (rule 8: never baked into the image). Only a few episodes (data JSON + 3 RGB + 3 depth mp4)
# are pulled so the laptop/remote stay lean; enough for open-loop replay + MAE.
#   ryzers run /ryzers/scripts/download_datasets.sh [robotwin|robocasa]
# Env: CHUNK=chunk-0000  EP_GLOB=episode_000000[0-4]  (fnmatch pattern for the episode subset)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HF_COMMON:-$HERE/_hf_common.sh}"

WHICH="${1:-robotwin}"
CHUNK="${CHUNK:-chunk-0000}"
EP_GLOB="${EP_GLOB:-episode_000000[0-4]}"
DATASET_ROOT="${DATASET_ROOT:-/models/xwam/datasets}"

case "$WHICH" in
  robotwin) REPO="sharinka0715/X-WAM-RoboTwin"; DEST="$DATASET_ROOT/RoboTwin" ;;
  robocasa) REPO="sharinka0715/X-WAM-RoboCasa"; DEST="$DATASET_ROOT/RoboCasa" ;;
  *) echo "usage: download_datasets.sh [robotwin|robocasa]" >&2; exit 2 ;;
esac
mkdir -p "$DEST"

hf_prefetch "$REPO" --repo-type dataset --local-dir "$DEST" \
  --include "data/${CHUNK}/${EP_GLOB}.json" \
            "video/*/${CHUNK}/${EP_GLOB}.mp4" \
            "depth/*/${CHUNK}/${EP_GLOB}.mp4"

echo "PASS: X-WAM $WHICH open-loop subset ($CHUNK/$EP_GLOB) cached under $DEST"
find "$DEST" -name '*.mp4' | head -n 12
echo "... ($(find "$DEST" -name '*.mp4' | wc -l) mp4 files, $(find "$DEST" -name '*.json' | wc -l) episode json)"
