#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Interactive LIBERO demo (synchronous chunk-replay) driven by the FLUX.2 ImageWAM policy,
# served over HTTP/MJPEG. Requires the simulation/libero base:
#   ryzers build libero imagewam
#   ryzers run /ryzers/demos/demo_interactive_libero.sh
# View at http://localhost:PORT (ssh -L PORT:localhost:PORT <host>).
set -euo pipefail
if [ ! -d /opt/sim/sim_libero ]; then
  echo "ERROR: simulation/libero base not found (no /opt/sim/sim_libero)." >&2
  echo "       Build the chain:  ryzers build libero imagewam" >&2
  exit 1
fi
REPO="${IMAGEWAM_REPO:-/repos/imagewam}"
FLUX2_SRC="${FLUX2_SRC:-/repos/flux2}"
VARIANT="${FLUX2_VARIANT:-4b}"
export SUITE="${SUITE:-libero_object}"
export TASK_ID="${TASK_ID:-0}"
export SEED="${SEED:-1000}"
export PORT="${PORT:-8080}"
export OUT_DIR="${OUT_DIR:-/outputs}"
export CKPT="${CKPT:-/models/imagewam_release/libero/flux2_klein_${VARIANT}/model.pt}"
export DATASET_STATS="${DATASET_STATS:-/models/imagewam_release/libero/flux2_klein_${VARIANT}/dataset_stats.json}"
export FLUX2_SRC FLUX2_VARIANT="${VARIANT}"
export FLUX2_MODEL_PATH="${FLUX2_MODEL_PATH:-/models/flux2/FLUX.2-klein-base-4B/flux-2-klein-base-4b.safetensors}"
export FLUX2_AE_MODEL_PATH="${FLUX2_AE_MODEL_PATH:-/models/flux2/FLUX.2-klein-base-4B/ae.safetensors}"
export FLUX2_QWEN3_MODEL_SPEC="${FLUX2_QWEN3_MODEL_SPEC:-Qwen/Qwen3-4B}"
export REPLAN_STEPS="${REPLAN_STEPS:-12}"
export POLICY_FACTORY="imagewam_libero_policy:build_policy"
export PYTHONPATH="/opt/imagewam-adapters:${REPO}:${REPO}/src:${FLUX2_SRC}/src:${FLUX2_SRC}:${REPO}/experiments/libero:/opt/sim:/opt/LIBERO:${PYTHONPATH:-}"

for f in "$CKPT" "$DATASET_STATS" "$FLUX2_MODEL_PATH" "$FLUX2_AE_MODEL_PATH"; do
  [ -f "$f" ] || { echo "missing weight: $f" >&2; exit 1; }
done
exec python -m sim_libero.interactive_server
