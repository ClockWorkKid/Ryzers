#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Closed-loop Meta-World entrypoint. One AVDC closed-loop rollout (video model -> UniMatch flow ->
# actions) in the MuJoCo simulator (headless EGL on the iGPU); writes the executed rollout GIF +
# metrics to $OUT_DIR/metaworld. Needs the MW checkpoint (download_checkpoints.sh WHICH=metaworld);
# UniMatch flow weights are baked in.
#   ENV_NAME  MW V2 task (default door-open-v2-goal-observable)  SEED  CAMERA(corner)
#   SAMPLE_STEPS(20)  MAX_REPLANS(5)  CKPT_DIR(/models/metaworld)  MILESTONE(24)
set -euo pipefail
: "${CKPT_DIR:=/models/metaworld}"
: "${MILESTONE:=24}"
export CKPT_DIR MILESTONE
AVDC_REPO="${AVDC_REPO:-/repos/avdc}"
cd "$AVDC_REPO/experiment"   # name2maskid.json + pretrained/ (flow weights) resolve here
python "${DEMO_PY:-/ryzers/demos/demo_metaworld.py}"
