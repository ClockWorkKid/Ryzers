#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Fetch the released VERA checkpoints (sizhe-lester-li/VERA) into the mounted model volume.
# The frozen upstream bases (Wan-AI/Wan2.1-T2V-1.3B, Wan-AI/Wan2.1-I2V-14B-480P, facebook/VGGT-1B)
# are pulled separately by the upstream loaders on first model run (rule 8).
#   ryzers run /ryzers/scripts/download_checkpoints.sh [wave1|droid|all]
#     wave1 : MimicGen + PushT planners/IDMs        (~15 GB, Wave-1 notebooks)
#     droid : DROID WAN 14B planner + demo clips     (~31 GB, droid_generation.ipynb)
#     all   : everything incl. the 33 GB OMNI planner (~73 GB)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HF_COMMON:-$HERE/_hf_common.sh}"

WHICH="${1:-wave1}"
DEST="${VERA_CKPT_ROOT:-/models/vera-ckpts}"
mkdir -p "$DEST"

REPO="sizhe-lester-li/VERA"
case "$WHICH" in
  wave1)
    hf_prefetch "$REPO" --local-dir "$DEST" \
      --include "mimicgen-wan-1.3b/*" "idm-mimicgen-285ouq1q/*" "pusht-dfot/*" "pusht-idm/*" ;;
  droid)
    hf_prefetch "$REPO" --local-dir "$DEST" \
      --include "wan-droid-14b/*" "droid-demo-clips/*" ;;
  all)
    hf_prefetch "$REPO" --local-dir "$DEST" ;;
  *) echo "usage: download_checkpoints.sh [wave1|droid|all]" >&2; exit 2 ;;
esac

echo "PASS: VERA checkpoints ($WHICH) cached under $DEST"
ls -la "$DEST"
