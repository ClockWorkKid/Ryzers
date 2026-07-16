#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Fetch a few episodes of the Cosmos3-DROID dataset (nvidia/Cosmos3-DROID, LeRobotDataset v3.0)
# for the open-loop eval. GATED: export HF_TOKEN + accept the license (rule 8: fetch, never
# re-host). The eval loader points DROID_ROOT at the resulting `.../success` dir (must contain
# meta/info.json).
#   NUM_EPISODES=5 ryzers run /ryzers/scripts/download_datasets.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HF_COMMON:-$HERE/_hf_common.sh}"

REPO="${COSMOS_DROID_REPO:-nvidia/Cosmos3-DROID}"
DEST="${COSMOS_DROID_DIR:-/models/cosmos3_droid}"
mkdir -p "$DEST"

# NOTE (Phase 3): confirm the repo's directory layout (success/meta, success/data/chunk-*),
# then narrow --include to only the first NUM_EPISODES episodes to keep the download small.
# Full-repo fetch is large; prefer a targeted --include glob once the layout is verified.
hf_prefetch "$REPO" --repo-type dataset --local-dir "$DEST" --include "success/meta/*"
echo "PASS: Cosmos3-DROID (meta) cached under $DEST"
echo "NOTE: extend --include with success/data/chunk-*/episode_00000{0..N} for actual episodes."
ls -la "$DEST" || true
