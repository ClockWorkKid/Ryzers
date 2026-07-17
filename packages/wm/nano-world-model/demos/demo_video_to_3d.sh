#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Application: video -> 3D point cloud via Depth Anything 3 (upstream src/scripts/video_to_pointcloud.py).
# Feeds a generated rollout video into DA3 multi-view depth estimation -> colored PLY point cloud
# + per-frame depth-visualization PNGs (original | depth | confidence) as the inspectable artifact.
#
# Best effort: DA3 + viser are installed as a soft-fail image layer; if the import check fails this
# script exits 0 with a note so the overnight orchestrator continues.
set -euo pipefail

REPO=/repos/nanowm
OUT_DIR="${OUT_DIR:-/outputs}"
VIDEO="${VIDEO:-$OUT_DIR/rollout_dino_wm_pusht/sample_0000_gen.mp4}"
OUT_PLY="${OUT_PLY:-$OUT_DIR/video_to_3d/scene.ply}"
MODEL="${DA3_MODEL:-depth-anything/DA3-LARGE-1.1}"
MAX_FRAMES="${MAX_FRAMES:-16}"

if ! python -c "import depth_anything_3" 2>/dev/null; then
  echo "[v3d] Depth-Anything-3 not importable in this image; skipping (best-effort app)."
  exit 0
fi
if [ ! -f "$VIDEO" ]; then
  echo "[v3d] input video not found: $VIDEO (run a rollout demo first); skipping."
  exit 0
fi

mkdir -p "$(dirname "$OUT_PLY")"
cd "$REPO"
echo "[v3d] $VIDEO -> $OUT_PLY  (model=$MODEL)"
START=$(date +%s.%N)
python src/scripts/video_to_pointcloud.py \
  --video "$VIDEO" \
  --output "$OUT_PLY" \
  --model "$MODEL" \
  --max_frames "$MAX_FRAMES" \
  --device cuda \
  --save_depth_vis ${NATIVE_RES:+--native_res $NATIVE_RES}
END=$(date +%s.%N)
echo "[v3d] done in $(python -c "print(f'{${END}-${START}:.1f}')")s"
ls -la "$(dirname "$OUT_PLY")" 2>/dev/null || true
