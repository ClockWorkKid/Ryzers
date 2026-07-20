#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Phase 3 closed-loop Meta-World entrypoint. Runs one AVDC closed-loop rollout (video model ->
# UniMatch flow -> actions) in the MuJoCo simulator (headless EGL on the iGPU) and writes the
# executed rollout GIF + metrics to $OUT_DIR/metaworld. Needs the MW checkpoint
# (scripts/download_checkpoints.sh WHICH=metaworld); the UniMatch flow weights are baked in.
#
# Knobs (VAR=... ryzers run --name avdc /ryzers/demos/demo_metaworld.sh):
#   ENV_NAME      Meta-World V2 goal-observable task   (default door-open-v2-goal-observable)
#   SEED          episode seed                         (default 0)
#   CAMERA        corner | corner2 | corner3           (default corner)
#   SAMPLE_STEPS  DDIM steps per replan                (default 20)
#   MAX_REPLANS   max video re-plans before giving up  (default 5)
#   CKPT_DIR/MILESTONE  snapshot selection             (default /models/metaworld, 24)
# Runs from the experiment/ dir so name2maskid.json + pretrained/ (flow weights) resolve.
set -euo pipefail

AVDC_REPO="${AVDC_REPO:-/repos/avdc}"
cd "$AVDC_REPO/experiment"
python "${DEMO_PY:-/ryzers/demos/demo_metaworld.py}"
