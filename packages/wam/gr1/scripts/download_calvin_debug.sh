#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Fetch the small CALVIN *debug* dataset (~1.3 GB) into the mounted data dir at runtime (rule 8:
# not re-hosted; direct upstream link from the CALVIN benchmark). This subset is enough for the
# offline open-loop replay (demo_openloop) and the closed-loop debug eval (demo_calvin) — the full
# task_ABCD_D (~500 GB) is intentionally NOT downloaded (rules 3/4). Real episodes provide
# rgb_static / rgb_gripper / robot_obs / rel_actions + language annotations.
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/data}"
URL="http://calvin.cs.uni-freiburg.de/dataset/calvin_debug_dataset.zip"
DEST="$DATA_ROOT/calvin_debug_dataset"

mkdir -p "$DATA_ROOT"
if [ -d "$DEST" ]; then
  echo "exists, skip -> $DEST"; ls -1 "$DEST"; exit 0
fi

echo "Downloading $URL -> $DATA_ROOT/calvin_debug.zip"
if command -v wget >/dev/null 2>&1; then wget -q --show-progress -O "$DATA_ROOT/calvin_debug.zip" "$URL";
else curl -fL --progress-bar -o "$DATA_ROOT/calvin_debug.zip" "$URL"; fi

echo "Extracting ..."
python -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" \
  "$DATA_ROOT/calvin_debug.zip" "$DATA_ROOT"
rm -f "$DATA_ROOT/calvin_debug.zip"
echo "done -> $DEST"
ls -1 "$DEST"
