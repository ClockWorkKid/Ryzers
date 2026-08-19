#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Video imagination: denoise AHA-WAM's world-model video branch from a ground-truth
# RoboTwin 2.0 start frame + instruction and write side-by-side GT-vs-imagined clips to
# /outputs/videogen/videogen_<TAG>. Runs on the standalone model image (no simulator).
#   ryzers run /ryzers/demos/demo_videogen.sh
#   NUM_VIDEOS=6 NUM_STEPS=20 ryzers run /ryzers/demos/demo_videogen.sh
set -euo pipefail
DATASET="${DATASET:-robotwin}"
[ "$DATASET" = "robotwin" ] || { echo "AHA-WAM ships only RoboTwin 2.0 weights; DATASET must be robotwin" >&2; exit 2; }
REL=/models/ahawam_release
export AHAWAM_REPO=/repos/ahawam
export PYTHONPATH="/repos/ahawam/src:/repos/ahawam:${PYTHONPATH:-}"
export NUM_VIDEOS="${NUM_VIDEOS:-10}"
export NUM_STEPS="${NUM_STEPS:-20}"
export OUT_DIR="${OUT_DIR:-/outputs}/videogen"

export CONFIG_NAME=sim_robotwin TAG="${TAG:-robotwin_ahawam}"
export CKPT="${CKPT:-$REL/robotwin_ahawam.pt}"
export DATASET_STATS="${DATASET_STATS:-$REL/dataset_stats.json}"
export DATASET_DIR="${DATASET_DIR:-/models/data/robotwin2.0}"

# Ensure checkpoint + dataset are present (idempotent; resume from cache).
bash /ryzers/scripts/download_checkpoints.sh robotwin
{ [ -d "$DATASET_DIR" ] && [ -n "$(ls -A "$DATASET_DIR" 2>/dev/null)" ]; } || bash /ryzers/scripts/download_datasets.sh robotwin

[ -f "$CKPT" ] || { echo "missing $CKPT -> run scripts/download_checkpoints.sh robotwin" >&2; exit 1; }
[ -d "$DATASET_DIR" ] || { echo "missing $DATASET_DIR -> run scripts/download_datasets.sh robotwin" >&2; exit 1; }

exec python /ryzers/scripts/videogen_joint.py
