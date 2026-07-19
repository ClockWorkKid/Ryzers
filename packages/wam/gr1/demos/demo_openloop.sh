#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Phase 2 open-loop demo entrypoint. Loads real MAE + GR-1 weights and replays a real CALVIN
# episode window offline (no simulator), writing an action overlay (GT vs pred) + a future-frame
# two-column comparison to $OUT_DIR/openloop. Weights: scripts/download_checkpoints.sh (WHICH=all).
# Data: the CALVIN debug dataset mounted at $DATASET_DIR (default /data/calvin_debug_dataset).
#
# Knobs (override from host: VAR=... ryzers run --name gr1 /ryzers/demos/demo_openloop.sh):
#   WINDOW_IDX   language-annotation window index         (default 0)
#   MAX_STEPS    open-loop steps to roll                  (default 32)
#   SPLIT        training|validation                      (default validation)
#   SEED         reproducible seed                        (default 0)
set -euo pipefail

GR1_REPO="${GR1_REPO:-/repos/gr1}"
args=()
[ -n "${WINDOW_IDX:-}" ] && args+=(--window-idx "$WINDOW_IDX")
[ -n "${MAX_STEPS:-}" ]  && args+=(--max-steps "$MAX_STEPS")
[ -n "${SPLIT:-}" ]      && args+=(--split "$SPLIT")
[ -n "${SEED:-}" ]       && args+=(--seed "$SEED")
[ -n "${DATASET_DIR:-}" ] && args+=(--data-dir "$DATASET_DIR")

cd "$GR1_REPO"
python "${DEMO_PY:-/ryzers/demos/demo_openloop.py}" "${args[@]}"
