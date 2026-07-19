#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# MimicGen simulator sanity: headless RandomPolicy rollout -> MP4 (no model, no server).
# Steps robosuite/MuJoCo, resets to demo initial states from the task hdf5, renders offscreen
# via EGL (osmesa software fallback), and writes per-view videos under $OUT_DIR. Proves the
# simulator harness is alive end to end. Fetch a task first:
#   ryzers run --name sim-mimicgen /ryzers/scripts/download_mimicgen_datasets.sh stack_d0
#   ryzers run --name sim-mimicgen /ryzers/demos/demo_sim_sanity.sh
set -uo pipefail

: "${TASK:=stack_d0}"
: "${MIMICGEN_DATASET_ROOT:=/sim_data/mimicgen_datasets}"
: "${DATASET:=$MIMICGEN_DATASET_ROOT/core/${TASK}.hdf5}"
: "${OUT_DIR:=/sim_outputs/mimicgen/${TASK}_sanity}"
: "${NUM_DEMOS:=1}"
: "${ROLLOUT_HORIZON:=60}"
: "${RENDER_SIZE:=128}"
mkdir -p "$OUT_DIR"

if [ ! -f "$DATASET" ]; then
  echo "[demo_sim_sanity] dataset not found: $DATASET"
  echo "[demo_sim_sanity] fetch it first: ryzers run /ryzers/scripts/download_mimicgen_datasets.sh $TASK"
  exit 1
fi

echo "[demo_sim_sanity] TASK=$TASK DATASET=$DATASET policy=${POLICY_FACTORY:-RandomPolicy}"
run_client() {  # $1 = GL backend
  echo "[demo_sim_sanity] rollout with MUJOCO_GL=$1"
  MUJOCO_GL="$1" PYOPENGL_PLATFORM="$1" python -u -m sim_mimicgen.sanity \
    --dataset "$DATASET" --output-dir "$OUT_DIR" \
    --num-demos "$NUM_DEMOS" --rollout-horizon "$ROLLOUT_HORIZON" --render-size "$RENDER_SIZE" \
    ${VIEWS:+--views "$VIEWS"} ${CONTEXT_FRAMES:+--context-frames "$CONTEXT_FRAMES"} \
    --run-tag sanity
}
run_client egl; RC=$?
if [ "$RC" -ne 0 ]; then
  echo "[demo_sim_sanity] EGL path failed (rc=$RC); retrying with software osmesa"
  run_client osmesa; RC=$?
fi
echo "[demo_sim_sanity] videos under $OUT_DIR"
exit $RC
