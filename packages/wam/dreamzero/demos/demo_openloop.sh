#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Open-loop numeric validation: teacher-force released DROID episodes, predict
# one 24-step action chunk per anchor and score arm RMSE / per-step cosine /
# gripper accuracy vs ground truth. Writes overlays + aggregate.json to
# /outputs. First run needs the checkpoint + a few eval episodes.
#   EPISODES=0,1 CHUNKS_PER_EPISODE=3 ryzers run /ryzers/demos/demo_openloop.sh
set -euo pipefail

MODEL_DIR="${MODEL_PATH:-/models/DreamZero-DROID}"
DATA_DIR="${DROID_DATASET_DIR:-/models/DreamZero-DROID-Data}"
[ -f "$MODEL_DIR/config.json" ] || { echo "missing $MODEL_DIR -> run scripts/download_checkpoints.sh model" >&2; exit 1; }
[ -d "$DATA_DIR/meta" ]        || { echo "missing $DATA_DIR  -> run scripts/download_checkpoints.sh data"  >&2; exit 1; }

export EPISODES="${EPISODES:-0}"
export CHUNKS_PER_EPISODE="${CHUNKS_PER_EPISODE:-3}"
export OUTPUT_DIR="${OUTPUT_DIR:-/outputs}/openloop"
mkdir -p "$OUTPUT_DIR"

exec python /wam-direct/overlay/tests/validate_stage3_path_b.py
