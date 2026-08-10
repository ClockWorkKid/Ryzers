#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Fetch the DreamZero-DROID 14B checkpoint + its Wan2.1 base + a small slice of
# the released DROID eval data into the mounted volumes. Idempotent (resumes
# from cache). Big: DreamZero-DROID ~28 GB, Wan2.1-I2V-14B ~40 GB.
#   ryzers run /ryzers/scripts/download_checkpoints.sh [model|data|all]
#   EPISODES_N=3 ryzers run /ryzers/scripts/download_checkpoints.sh data
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HF_COMMON:-$HERE/_hf_common.sh}"

WHICH="${1:-all}"
MODEL_DIR="${MODEL_PATH:-/models/DreamZero-DROID}"
DATA_DIR="${DROID_DATASET_DIR:-/models/DreamZero-DROID-Data}"
WAN_DIR="${WAN21_DIR:-/root/.cache/huggingface/Wan-AI/Wan2.1-I2V-14B-480P}"
EPISODES_N="${EPISODES_N:-3}"   # how many episodes of eval data to pull

fetch_model() {
  hf_prefetch GEAR-Dreams/DreamZero-DROID --local-dir "$MODEL_DIR"
  # Wan2.1 base: --local-dir gives the DiffSynth flat layout (WAN21_DIR: models_t5,
  # models_clip, diffusion shards) and also populates the hub cache blobs that the
  # backbone loader resolves by repo id.
  hf_prefetch Wan-AI/Wan2.1-I2V-14B-480P --local-dir "$WAN_DIR"
  # umt5-xxl text-encoder tokenizer/config (hub cache).
  hf_prefetch google/umt5-xxl
  echo "PASS: model + base cached (model=$MODEL_DIR wan=$WAN_DIR)"
}

fetch_data() {
  # Released eval data (LeRobot per-episode parquet + per-camera mp4 + meta).
  # Pull meta + the first EPISODES_N episodes only (full set is very large).
  local inc=(--include "meta/*")
  local i n
  for ((i=0; i<EPISODES_N; i++)); do
    n=$(printf "%06d" "$i")
    inc+=(--include "data/chunk-000/episode_${n}.parquet")
    inc+=(--include "videos/chunk-000/*/episode_${n}.mp4")
  done
  hf_prefetch GEAR-Dreams/DreamZero-DROID-Data --repo-type dataset \
    "${inc[@]}" --local-dir "$DATA_DIR"
  echo "PASS: DROID eval data (first $EPISODES_N episodes) cached under $DATA_DIR"
}

case "$WHICH" in
  model) fetch_model ;;
  data)  fetch_data ;;
  all)   fetch_model; fetch_data ;;
  *) echo "usage: download_checkpoints.sh [model|data|all]" >&2; exit 2 ;;
esac
