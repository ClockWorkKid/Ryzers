#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# iTHOR simulator sanity: headless ScriptedPolicy ObjectNav rollout -> video (no model).
# Brings up the ai2thor CloudRendering (Vulkan) controller on gfx1151, resets to a random reachable
# pose, steps discrete moves, renders offscreen, and writes a rollout video under $OUT_DIR. Proves
# the simulator harness is alive end to end. Select a real model with POLICY_FACTORY=module:function.
#   ryzers run --name sim-ithor /ryzers/demos/demo_sim_sanity.sh
set -uo pipefail

: "${SCENE:=FloorPlan1}"
: "${TARGET:=Toaster}"
: "${N_SEEDS:=1}"
: "${RESOLUTION:=64}"
: "${MAX_EPLEN:=50}"
: "${OUT_DIR:=/sim_outputs}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUT_DIR/ithor/${SCENE}_${TARGET}_sanity}"
mkdir -p "$OUTPUT_DIR"

echo "[demo_sim_sanity] SCENE=$SCENE TARGET=$TARGET policy=${POLICY_FACTORY:-ScriptedPolicy}"
echo "[demo_sim_sanity] platform=${THOR_PLATFORM:-CloudRendering} VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-(unset)}"

python -u -m sim_ithor.sanity \
  --scene "$SCENE" --target "$TARGET" --output-dir "$OUTPUT_DIR" \
  --n-seeds "$N_SEEDS" --resolution "$RESOLUTION" --max-eplen "$MAX_EPLEN" \
  ${RENDER_RESOLUTION:+--render-resolution "$RENDER_RESOLUTION"}
RC=$?
echo "[demo_sim_sanity] output under $OUTPUT_DIR (rc=$RC)"
exit $RC
