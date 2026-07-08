#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Open-loop replay: feed ground-truth observations from released RoboTwin 2.0 LeRobot
# episodes and overlay predicted vs GT action horizons (two-phase: one video-context
# prefill + rolled action chunks). Writes per-episode overlays, aggregate MAE graphs
# and summary.json to /outputs/openloop/openloop_<TAG>.
#   ryzers run /ryzers/demos/demo_openloop.sh
#   NUM_EPISODES=20 NUM_STEPS=10 ryzers run /ryzers/demos/demo_openloop.sh
set -euo pipefail
REL=/models/ahawam_release
export AHAWAM_REPO=/repos/ahawam
export PYTHONPATH="/repos/ahawam/src:/repos/ahawam:${PYTHONPATH:-}"
export NUM_EPISODES="${NUM_EPISODES:-6}"
export NUM_STEPS="${NUM_STEPS:-10}"
export OUT_DIR="${OUT_DIR:-/outputs}/openloop"

export CONFIG_NAME=sim_robotwin TAG="${TAG:-robotwin_ahawam}"
export CKPT="${CKPT:-$REL/robotwin_ahawam.pt}"
export DATASET_STATS="${DATASET_STATS:-$REL/dataset_stats.json}"
export DATASET_DIR="${DATASET_DIR:-/models/data/robotwin2.0}"

# Ensure checkpoint + dataset are present (idempotent; resume from cache).
bash /ryzers/scripts/download_checkpoints.sh robotwin
[ -d "$DATASET_DIR" ] || bash /ryzers/scripts/download_datasets.sh robotwin

[ -f "$CKPT" ] || { echo "missing $CKPT -> run scripts/download_checkpoints.sh robotwin" >&2; exit 1; }
[ -d "$DATASET_DIR" ] || { echo "missing $DATASET_DIR -> run scripts/download_datasets.sh robotwin" >&2; exit 1; }

exec python /ryzers/scripts/openloop_replay.py
