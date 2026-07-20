#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Closed-loop Meta-World benchmark: all (or TASKS=...) tasks x N_SEEDS in one process with the
# optimized fp16+compile config; writes per-task success + summary.json + one GIF/task to
# $OUT_DIR/benchmark. Long-running -> launch detached and poll (rule 0.5).
#   TASKS(all|list)  N_SEEDS(5)  CAMERA(corner)  SAMPLE_STEPS(10)  MAX_REPLANS(5)
set -euo pipefail
: "${CKPT_DIR:=/models/metaworld}"
: "${MILESTONE:=24}"
export CKPT_DIR MILESTONE
AVDC_REPO="${AVDC_REPO:-/repos/avdc}"
cd "$AVDC_REPO/experiment"
python "${DEMO_PY:-/ryzers/demos/demo_metaworld_benchmark.py}"
