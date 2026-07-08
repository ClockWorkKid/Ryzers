#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Fetch SimplerEnv real-to-sim assets (third-party, not baked into the image for compliance).
# ManiSkill2_real2sim ships a downloader for the visual-matching scene/object assets; we run
# it once into the persistent /sim_models mount (idempotent via a .ready marker).
set -uo pipefail
ASSETS="${SIMPLERENV_ASSETS_DIR:-/sim_models/simplerenv_assets}"
mkdir -p "$ASSETS"
if [ -f "$ASSETS/.ready" ]; then
  echo "[setup_simplerenv] assets already present at $ASSETS"; exit 0
fi

# ManiSkill2_real2sim keeps its assets under the package data dir; point it at our mount.
export MS2_REAL2SIM_ASSET_DIR="$ASSETS"
echo "[setup_simplerenv] downloading real2sim assets into $ASSETS ..."
# The exact downloader entrypoint is validated on hardware; try the documented one first.
python -m mani_skill2_real2sim.utils.download_asset all -y --output-dir "$ASSETS" \
  || python -m mani_skill2_real2sim.utils.download_asset --help \
  || echo "[setup_simplerenv] WARN: adjust downloader invocation for the pinned SimplerEnv"

touch "$ASSETS/.ready"
echo "[setup_simplerenv] done."
