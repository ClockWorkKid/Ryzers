#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Runtime asset fetch for RoboLab-AMD (rules 3 & 8: never bake third-party binaries).
#
# The pilot task (BananaInBowl) uses the REAL YCB 011_banana + 024_bowl meshes -- the same
# source assets upstream RoboLab uses. This fetches them from the open YCB benchmark S3
# bucket and convex-decomposes (coacd) each into a robosuite-loadable MJCF, into the mounted
# assets volume $ROBOLAB_ASSETS_DIR. Idempotent: skips objects already converted. The env
# also does this lazily on first build, so running this explicitly is optional (it just makes
# the first rollout faster and surfaces download errors early).
set -euo pipefail
ASSETS_DIR="${ROBOLAB_ASSETS_DIR:-/models/robolab_assets}"
mkdir -p "$ASSETS_DIR"
echo "[setup_robolab] fetching + converting YCB banana/bowl into $ASSETS_DIR ..."
python -m sim_robolab.ycb_to_mjcf
echo "[setup_robolab] done."
