#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Phase 3b closed-loop Meta-World benchmark entrypoint. Runs all (or TASKS=...) tasks x N_SEEDS in
# one process with the optimized fp16+compile config; writes per-task success rates + summary.json
# + one sample GIF/task to $OUT_DIR/benchmark. Long-running -> launch detached and poll (rule 0.5).
#
# Knobs: TASKS(all|comma list) N_SEEDS(5) CAMERA(corner) SAMPLE_STEPS(10) MAX_REPLANS(5).
set -euo pipefail
AVDC_REPO="${AVDC_REPO:-/repos/avdc}"
cd "$AVDC_REPO/experiment"
python "${DEMO_PY:-/ryzers/demos/demo_benchmark.py}"
