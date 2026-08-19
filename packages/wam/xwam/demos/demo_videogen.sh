#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Depth imagination (X-WAM's unique joint RGB+depth path): denoise the future RGB latents AND
# the depth-modality latents from one GT start frame, VAE-decode per view, and write a GIF/MP4
# stacking imagined RGB, imagined depth, and GT depth to /outputs/imagination_<TAG>. Runs on
# the standalone xwam image (no simulator). First run fetches the SFT checkpoint + a small
# dataset subset (RGB + depth mp4s).
#   DATASET=robotwin ryzers run --name xwam /ryzers/demos/demo_videogen.sh
#   DATASET=robocasa NUM_VIDEOS=2 ryzers run --name xwam /ryzers/demos/demo_videogen.sh
set -euo pipefail

DATASET="${DATASET:-robotwin}"
EXP="${EXP:-${DATASET}_sft}"
bash /ryzers/scripts/download_checkpoints.sh "${EXP%_sft}"
bash /ryzers/scripts/download_datasets.sh "$DATASET"

export EXP TAG="${TAG:-$DATASET}"
export DATASET_ROOT="${DATASET_ROOT:-/models/xwam/datasets/$([ "$DATASET" = robocasa ] && echo RoboCasa || echo RoboTwin)}"
export NUM_VIDEOS="${NUM_VIDEOS:-2}"
export DENOISE_STEPS="${DENOISE_STEPS:-40}"
export ACTION_DENOISE_STEPS="${ACTION_DENOISE_STEPS:-10}"
exec python -u /ryzers/scripts/videogen_depth.py
