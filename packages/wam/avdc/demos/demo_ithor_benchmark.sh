#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# AVDC iTHOR ObjectNav benchmark: drive the sim base's multi-task benchmark driver with the AVDC
# policy (single process, so the fp16 + torch.compile warmup is amortized across all tasks).
# Writes result_dict.json + summary.json + per-task sample videos under $OUT_DIR/ithor/benchmark.
#   ryzers run --name avdc-ithor /ryzers/scripts/download_checkpoints.sh
#   TASKS=all N_SEEDS=20 ryzers run --name avdc-ithor /ryzers/demos/demo_benchmark.sh
set -uo pipefail

: "${TASKS:=all}"
: "${N_SEEDS:=20}"
: "${CKPT_DIR:=/models/ithor}"
: "${MILESTONE:=16}"
export POLICY_FACTORY="${POLICY_FACTORY:-avdc_ithor_policy:build_policy}"
# The multi-task driver ships with the simulation/ithor base (present in this chained image).
: "${DEMO_PY:=/ryzers/demos/demo_benchmark.py}"

echo "[demo_benchmark] TASKS=$TASKS N_SEEDS=$N_SEEDS policy=$POLICY_FACTORY ckpt=$CKPT_DIR/model-$MILESTONE.pt"
echo "[demo_benchmark] AMP=${AVDC_AMP:-off} COMPILE=${AVDC_COMPILE:-off} SAMPLE_STEPS=${SAMPLE_STEPS:-default}"
python -u "$DEMO_PY" \
  --tasks "$TASKS" --n-seeds "$N_SEEDS" \
  ${MAX_EPLEN:+--max-eplen "$MAX_EPLEN"} \
  ${RENDER_RESOLUTION:+--render-resolution "$RENDER_RESOLUTION"} \
  --out "${OUT_DIR:-/outputs}/ithor/benchmark"
