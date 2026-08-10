#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Predicted-video (looping-fix) rollout: K-chunk continuous predicted video per
# episode with the open-loop looping fix ON:
#   * VIDEO_FPS=5              -- model's native predicted-video rate (15 plays
#                                ~3.2x too fast and *looks* like it loops)
#   * STREAMING_OVERLAP_DECODE -- rolling latent overlap at chunk boundaries so
#                                the VAE keeps temporal context (kills the seam)
#   * STAGE_C=0                -- no image-feature reuse at local-attn resets
# Plus an autoregressive extrapolation pass (long_video_autoregressive_debug.mp4).
# Writes per-episode mp4s + side-by-side (anchor|prediction) + manifest to /outputs.
#   EPISODES=0 NUM_CHUNKS=4 ryzers run /ryzers/demos/demo_videogen.sh
set -euo pipefail

MODEL_DIR="${MODEL_PATH:-/models/DreamZero-DROID}"
DATA_DIR="${DROID_DATASET_DIR:-/models/DreamZero-DROID-Data}"
[ -f "$MODEL_DIR/config.json" ] || { echo "missing $MODEL_DIR -> run scripts/download_checkpoints.sh model" >&2; exit 1; }
[ -d "$DATA_DIR/meta" ]        || { echo "missing $DATA_DIR  -> run scripts/download_checkpoints.sh data"  >&2; exit 1; }

export EPISODES="${EPISODES:-0}"
export NUM_CHUNKS="${NUM_CHUNKS:-4}"
export DENOISE_STEPS="${DENOISE_STEPS:-2}"
export VIDEO_FPS="${VIDEO_FPS:-5}"
export STAGE_C="${STAGE_C:-0}"
export STREAMING_OVERLAP_DECODE="${STREAMING_OVERLAP_DECODE:-1}"
export STREAMING_OVERLAP_K="${STREAMING_OVERLAP_K:-1}"
export LONG_VIDEO_AUTOREGRESSIVE_DEBUG="${LONG_VIDEO_AUTOREGRESSIVE_DEBUG:-1}"
export OUTPUT_DIR="${OUTPUT_DIR:-/outputs}/videogen"
mkdir -p "$OUTPUT_DIR"

exec python /wam-direct/overlay/tests/render_stage5_long.py
