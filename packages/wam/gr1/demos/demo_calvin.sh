#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Phase 3 closed-loop CALVIN eval entrypoint (headless PyBullet + EGL on gfx1151). Drives GR-1
# in the simulator on the CALVIN debug validation tasks and writes rollout GIFs + a success rate
# to $OUT_DIR/calvin. Requires the CALVIN sim layer in the image + the debug dataset at
# $DATASET_DIR and weights from scripts/download_checkpoints.sh (WHICH=all).
#
# Knobs (override from host: VAR=... ryzers run --name gr1 /ryzers/demos/demo_calvin.sh):
#   NUM_TASKS   number of annotated tasks to eval (0 = all 8 debug tasks)   (default 0)
#   EP_LEN      max closed-loop steps per task                              (default 120)
#   SPLIT       training|validation                                         (default validation)
#   SEED        reproducible seed                                           (default 0)
set -euo pipefail

# Headless EGL rendering on the AMD iGPU.
export MESA_GL_VERSION_OVERRIDE=3.3
export PYOPENGL_PLATFORM=egl
export CALVIN_ROOT="${CALVIN_ROOT:-/repos/calvin}"
# Reuse the CLIP weights already fetched into the mounted models dir (avoid re-download).
mkdir -p "$HOME/.cache"
[ -d /models/clip_cache ] && ln -sfn /models/clip_cache "$HOME/.cache/clip" || true

GR1_REPO="${GR1_REPO:-/repos/gr1}"
args=()
[ -n "${NUM_TASKS:-}" ] && args+=(--num-tasks "$NUM_TASKS")
[ -n "${EP_LEN:-}" ]    && args+=(--max-steps "$EP_LEN")
[ -n "${SPLIT:-}" ]     && args+=(--split "$SPLIT")
[ -n "${SEED:-}" ]      && args+=(--seed "$SEED")
[ -n "${DATASET_DIR:-}" ] && args+=(--data-dir "$DATASET_DIR")

cd "$GR1_REPO"
python "${DEMO_PY:-/ryzers/demos/demo_calvin.py}" "${args[@]}"
