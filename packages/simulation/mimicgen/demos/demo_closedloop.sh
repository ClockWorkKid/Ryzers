#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Generic model-driven MimicGen closed-loop (model-agnostic). Drives whatever policy
# POLICY_FACTORY selects: an in-process sim_mimicgen.Policy, or a remote websocket model server
# via sim_mimicgen.remote_policy:build_policy (set POLICY_HOST/POLICY_PORT to a *running* server).
# A chained model layer typically wraps this in its own demo that first starts its policy server
# (e.g. VERA's demo_closedloop_mimicgen.sh). Fetch a task first (download_mimicgen_datasets.sh).
#   POLICY_FACTORY=sim_mimicgen.remote_policy:build_policy POLICY_PORT=8800 \
#     TASK=stack_d0 ryzers run --name <model>-mimicgen /ryzers/demos/demo_closedloop.sh
set -uo pipefail

: "${TASK:=stack_d0}"
: "${MIMICGEN_DATASET_ROOT:=/sim_data/mimicgen_datasets}"
: "${DATASET:=$MIMICGEN_DATASET_ROOT/core/${TASK}.hdf5}"
: "${OUT_DIR:=/sim_outputs/mimicgen/${TASK}}"
: "${NUM_DEMOS:=3}"
: "${ROLLOUT_HORIZON:=400}"
: "${RENDER_SIZE:=128}"
mkdir -p "$OUT_DIR"

if [ ! -f "$DATASET" ]; then
  echo "[demo_closedloop] dataset not found: $DATASET"
  echo "[demo_closedloop] fetch it first: ryzers run /ryzers/scripts/download_mimicgen_datasets.sh $TASK"
  exit 1
fi
if [ -z "${POLICY_FACTORY:-}" ]; then
  echo "[demo_closedloop] WARNING: POLICY_FACTORY unset -> RandomPolicy (no model). Set it to your"
  echo "[demo_closedloop] adapter, or sim_mimicgen.remote_policy:build_policy (+POLICY_HOST/PORT)."
fi

echo "[demo_closedloop] TASK=$TASK DATASET=$DATASET policy=${POLICY_FACTORY:-RandomPolicy}"
run_client() {  # $1 = GL backend
  echo "[demo_closedloop] rollout with MUJOCO_GL=$1"
  MUJOCO_GL="$1" PYOPENGL_PLATFORM="$1" python -u -m sim_mimicgen.sanity \
    --dataset "$DATASET" --output-dir "$OUT_DIR" \
    --num-demos "$NUM_DEMOS" --rollout-horizon "$ROLLOUT_HORIZON" --render-size "$RENDER_SIZE" \
    ${VIEWS:+--views "$VIEWS"} ${CONTEXT_FRAMES:+--context-frames "$CONTEXT_FRAMES"} \
    --run-tag closedloop
}
run_client egl; RC=$?
if [ "$RC" -ne 0 ]; then
  echo "[demo_closedloop] EGL path failed (rc=$RC); retrying with software osmesa"
  run_client osmesa; RC=$?
fi
echo "[demo_closedloop] videos under $OUT_DIR"
exit $RC
