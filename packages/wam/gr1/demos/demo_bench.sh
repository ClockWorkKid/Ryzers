#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Phase 4 benchmark entrypoint: times GR-1's closed-loop step path on a real CALVIN episode and
# reports latency + action quality for the env-selected optimization config (GR1_AMP / GR1_SDPA /
# GR1_COMPILE; see demos/gr1_optim.py). Results -> $OUT_DIR/bench/bench_<GR1_TAG>.json.
set -euo pipefail
export CALVIN_ROOT="${CALVIN_ROOT:-/repos/calvin}"
mkdir -p "$HOME/.cache"
[ -d /models/clip_cache ] && ln -sfn /models/clip_cache "$HOME/.cache/clip" || true
cd "${GR1_REPO:-/repos/gr1}"
python "${DEMO_PY:-/ryzers/demos/demo_bench.py}" "$@"
