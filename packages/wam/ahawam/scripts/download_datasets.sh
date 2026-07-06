#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Fetch the preprocessed LeRobot RoboTwin 2.0 dataset used by the open-loop /
# video-imagination demos into the mounted model volume:
#   RoboTwin : yuanty/robotwin2.0-fastwam  (split archives, concatenated)
# (AHA-WAM is trained/evaluated on RoboTwin 2.0; there is no LIBERO checkpoint.)
#   ryzers run /ryzers/scripts/download_datasets.sh [robotwin]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HF_COMMON:-$HERE/_hf_common.sh}"

WHICH="${1:-robotwin}"
DATA="${AHAWAM_DATA_DIR:-/models/data}"
mkdir -p "$DATA"

fetch_robotwin() {
  local tmp="$DATA/.dl_robotwin"; mkdir -p "$tmp"
  hf_prefetch yuanty/robotwin2.0-fastwam --repo-type dataset --local-dir "$tmp"
  echo "concatenate split archives + extract"
  cat "$tmp"/robotwin2.0.tar.gz.part-* | tar -xzf - -C "$DATA"
  # Flatten if the archive nests robotwin2.0/robotwin2.0/.
  if [ -d "$DATA/robotwin2.0/robotwin2.0" ]; then
    mv "$DATA/robotwin2.0/robotwin2.0"/* "$DATA/robotwin2.0/" && rmdir "$DATA/robotwin2.0/robotwin2.0"
  fi
  [ -f "$tmp/dataset_stats.json" ] && cp "$tmp/dataset_stats.json" "$DATA/robotwin2.0/" || true
  rm -rf "$tmp"
  echo "PASS: RoboTwin dataset under $DATA/robotwin2.0"
}

case "$WHICH" in
  robotwin|all) fetch_robotwin ;;
  *) echo "usage: download_datasets.sh [robotwin]" >&2; exit 2 ;;
esac
ls -la "$DATA"
