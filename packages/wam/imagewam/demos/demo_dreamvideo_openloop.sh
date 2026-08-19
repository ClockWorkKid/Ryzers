#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Open-loop per-step "dream" VIDEO on REAL episodes (dataset replay, NO simulator). Steps one
# episode frame-by-frame; at every frame the model dreams its edited future (infer_video_flux2)
# and we stitch [ observation | GT future | dreamed future ] into one continuous MP4
# (ol_dreamvid_<dataset>.mp4 + first/mid/last still frames). video_augmentation is disabled so
# the observation/GT columns are deterministic replay (no per-sample crop/rotate/jitter shake).
# DATASET=libero (2-cam 224x448) | robotwin (3-cam compact). Needs the matching lerobot dataset.
#   DATASET=libero   DATA_DIR=/libero_data/libero_object_no_noops_lerobot \
#     ryzers run --name imagewam -v /host/libero:/libero_data /ryzers/demos/demo_dreamvideo_openloop.sh
#   DATASET=robotwin DATA_DIR=/robotwin_data/<task>_lerobot \
#     ryzers run --name imagewam -v /host/robotwin:/robotwin_data /ryzers/demos/demo_dreamvideo_openloop.sh
set -euo pipefail

DATASET="${DATASET:-libero}"
: "${DATA_DIR:?set DATA_DIR to the mounted lerobot dataset dir (and -v it into the container)}"
if [ ! -d "$DATA_DIR" ]; then
  echo "ERROR: $DATASET lerobot dataset not found at $DATA_DIR" >&2
  echo "       Datasets are shared with FastWAM: https://huggingface.co/datasets/yuanty" >&2
  exit 1
fi

bash /ryzers/scripts/download_checkpoints.sh "$DATASET" "${FLUX2_VARIANT:-4b}"

exec python /ryzers/scripts/dream_video_openloop.py
