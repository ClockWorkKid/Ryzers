#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# AVDC single closed-loop iTHOR ObjectNav rollout (headless ai2thor CloudRendering / Vulkan).
# Saves the executed rollout video + a rule-2.b two-column plan-vs-sim GIF + metrics.
#   ryzers run --name avdc-ithor /ryzers/scripts/download_checkpoints.sh
#   SCENE=FloorPlan1 TARGET=Toaster ryzers run --name avdc-ithor /ryzers/demos/demo_ithor.sh
set -uo pipefail

: "${SCENE:=FloorPlan1}"
: "${TARGET:=Toaster}"
: "${SEED:=0}"
: "${CKPT_DIR:=/models/ithor}"
: "${MILESTONE:=16}"

echo "[demo_ithor] SCENE=$SCENE TARGET=$TARGET seed=$SEED ckpt=$CKPT_DIR/model-$MILESTONE.pt"
echo "[demo_ithor] AMP=${AVDC_AMP:-off} COMPILE=${AVDC_COMPILE:-off} SAMPLE_STEPS=${SAMPLE_STEPS:-default}"
python -u "${DEMO_PY:-/ryzers/demos/demo_ithor.py}" \
  --scene "$SCENE" --target "$TARGET" --seed "$SEED" \
  ${SAMPLE_STEPS:+} ${MAX_EPLEN:+--max-eplen "$MAX_EPLEN"} \
  ${RENDER_RESOLUTION:+--render-resolution "$RENDER_RESOLUTION"}
