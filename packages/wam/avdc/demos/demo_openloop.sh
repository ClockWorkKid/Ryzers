#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Phase 2 open-loop video-prediction entrypoint. Loads a real AVDC snapshot + CLIP and generates
# the predicted future video from one initial frame + a task string (no simulator). Writes
# generated.gif + sidebyside.gif (initial frame | generated video, rule 2.b) to $OUT_DIR/openloop.
# Weights: scripts/download_checkpoints.sh (WHICH=metaworld). CLIP auto-downloads on first run.
#
# Knobs (override from host: VAR=... ryzers run --name avdc /ryzers/demos/demo_openloop.sh):
#   CKPT_DIR      snapshot dir (Trainer.load reads $CKPT_DIR/model-$MILESTONE.pt)  (default /models/metaworld)
#   MILESTONE     checkpoint milestone number                                      (default 24)
#   IMAGE         path to an initial frame (default: fetch upstream example)
#   TEXT          task/language string                                             (default 'assembly')
#   SAMPLE_STEPS  DDIM sampling steps (<=100)                                      (default 100)
#   FLOW          1 to use the optical-flow UNet variant                           (default 0)
#   SEED          reproducible seed                                                (default 0)
set -euo pipefail

AVDC_REPO="${AVDC_REPO:-/repos/avdc}"
cd "$AVDC_REPO/flowdiffusion"
python "${DEMO_PY:-/ryzers/demos/demo_openloop.py}"
