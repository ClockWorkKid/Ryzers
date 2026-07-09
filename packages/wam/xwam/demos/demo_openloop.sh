#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Open-loop replay: predict action chunks from released X-WAM episodes and overlay
# GT vs predicted actions + per-dim normalized MAE. First run downloads a small dataset
# subset + the SFT checkpoint. Usage:
#   DATASET=robotwin NUM_EPISODES=5 ryzers run --name xwam /ryzers/demos/demo_openloop.sh
set -euo pipefail

DATASET="${DATASET:-robotwin}"
EXP="${EXP:-${DATASET}_sft}"
bash /ryzers/scripts/download_checkpoints.sh "${EXP%_sft}"
bash /ryzers/scripts/download_datasets.sh "$DATASET"

export EXP TAG="${TAG:-$DATASET}"
export DATASET_ROOT="${DATASET_ROOT:-/models/xwam/datasets/$([ "$DATASET" = robocasa ] && echo RoboCasa || echo RoboTwin)}"
exec python /ryzers/scripts/openloop_replay.py
