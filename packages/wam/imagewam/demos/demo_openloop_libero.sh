#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# P5 open-loop evaluation on REAL LIBERO episodes (dataset replay, NO simulator): feeds real
# observations + proprio + task prompt to the model, compares predicted actions against the
# dataset's ground-truth future actions (MAE), the AE reconstruction (PSNR), and renders a few
# "dreamed" future frames. Writes ol_metrics.json, ol_action_overlay.png, ol_dream_*.png to
# /outputs. Runs entirely through ImageWAM's own dataset/processor + released dataset_stats so
# preprocessing/normalization match training. Needs the LIBERO lerobot dataset mounted.
#   ryzers run --name imagewam -v /host/libero:/libero_data /ryzers/demos/demo_openloop_libero.sh
#   LIBERO_SUITE_DIR=/libero_data/libero_object_no_noops_lerobot OL_NUM_SAMPLES=5 OL_NUM_DREAMS=3 \
#     ryzers run --name imagewam -v /host/libero:/libero_data /ryzers/demos/demo_openloop_libero.sh
set -euo pipefail

DATA="${LIBERO_SUITE_DIR:-/libero_data/libero_object_no_noops_lerobot}"
if [ ! -d "$DATA" ]; then
  echo "ERROR: LIBERO lerobot dataset not found at $DATA" >&2
  echo "       Mount it (-v /host/libero:/libero_data) or set LIBERO_SUITE_DIR. Datasets are" >&2
  echo "       shared with FastWAM: https://huggingface.co/datasets/yuanty/LIBERO-fastwam" >&2
  exit 1
fi

bash /ryzers/scripts/download_checkpoints.sh libero "${FLUX2_VARIANT:-4b}"

exec python /ryzers/scripts/open_loop_libero.py
