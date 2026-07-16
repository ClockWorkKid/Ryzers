#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Fetch the released Cosmos3-Nano-Policy-DROID checkpoint (nvidia/Cosmos3-Nano-Policy-DROID,
# ~32 GB+: transformer shards + vae + vision_encoder + tokenizers) into the mounted model
# volume / HF cache. GATED: export HF_TOKEN and accept the OpenMDW license on HF first (rule 8:
# never re-host weights; always fetch from upstream).
#   ryzers run /ryzers/scripts/download_checkpoints.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HF_COMMON:-$HERE/_hf_common.sh}"

REPO="${COSMOS_POLICY_REPO:-nvidia/Cosmos3-Nano-Policy-DROID}"
DEST="${COSMOS_RELEASE_DIR:-/models/cosmos3_policy_droid}"
mkdir -p "$DEST"

hf_prefetch "$REPO" --repo-type model --local-dir "$DEST"
echo "PASS: Cosmos3-Nano-Policy-DROID cached under $DEST"
ls -la "$DEST"
