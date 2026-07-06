#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Fetch the released AHA-WAM RoboTwin 2.0 checkpoints + dataset stats
# (SereneC/AHA-WAM-RoboTwin2.0) into the mounted model volume. AHA-WAM only ships
# RoboTwin2.0 weights (no LIBERO checkpoint). The Wan2.2 base is fetched separately
# by the upstream DiffSynth loader on the first model run (DIFFSYNTH_MODEL_BASE_PATH).
#   ryzers run /ryzers/scripts/download_checkpoints.sh [robotwin|flash|all]
#     robotwin -> robotwin_ahawam.pt        (AHA-WAM, ~24 Hz path)
#     flash    -> robotwin_ahawam-flash.pt  (ODE-distilled AHA-WAM-Flash, ~57 Hz path)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HF_COMMON:-$HERE/_hf_common.sh}"

WHICH="${1:-robotwin}"
DEST="${AHAWAM_RELEASE_DIR:-/models/ahawam_release}"
mkdir -p "$DEST"

files=(dataset_stats.json)
case "$WHICH" in
  robotwin) files+=(robotwin_ahawam.pt) ;;
  flash)    files+=(robotwin_ahawam-flash.pt) ;;
  all)      files+=(robotwin_ahawam.pt robotwin_ahawam-flash.pt) ;;
  *) echo "usage: download_checkpoints.sh [robotwin|flash|all]" >&2; exit 2 ;;
esac

hf_prefetch SereneC/AHA-WAM-RoboTwin2.0 "${files[@]}" --local-dir "$DEST"
echo "PASS: AHA-WAM checkpoints ($WHICH) cached under $DEST"
ls -la "$DEST"
