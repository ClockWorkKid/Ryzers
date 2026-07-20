#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# iTHOR ObjectNav benchmark: run the seam-selected policy across the tasks x seeds in one process.
# Model-agnostic - set POLICY_FACTORY to a model adapter (e.g. the AVDC consumer's) to benchmark it;
# with no POLICY_FACTORY it runs the built-in ScriptedPolicy (harness self-check, near-zero success).
#   TASKS=all N_SEEDS=20 POLICY_FACTORY=avdc_ithor_policy:build_policy \
#     ryzers run --name avdc-ithor /ryzers/demos/demo_benchmark.sh
set -uo pipefail

: "${TASKS:=all}"
: "${N_SEEDS:=20}"
: "${RESOLUTION:=64}"
: "${MAX_EPLEN:=50}"
: "${OUT_DIR:=/sim_outputs}"

echo "[demo_benchmark] TASKS=$TASKS N_SEEDS=$N_SEEDS policy=${POLICY_FACTORY:-ScriptedPolicy}"
python -u "${DEMO_PY:-/ryzers/demos/demo_benchmark.py}" \
  --tasks "$TASKS" --n-seeds "$N_SEEDS" --resolution "$RESOLUTION" --max-eplen "$MAX_EPLEN" \
  ${RENDER_RESOLUTION:+--render-resolution "$RENDER_RESOLUTION"}
