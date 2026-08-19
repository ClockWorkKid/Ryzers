#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Flow imagination: render FlowWAM's imagined future RGB alongside its PREDICTED optical-flow
# field (the model's unique dual-stream output), from a RoboTwin first-frame observation and
# the SAME dual-stream model + checkpoint as the closed loop. Writes
# imagination_<TAG>_flow.gif/.mp4 (imagined RGB | imagined flow-color) under /outputs/flowgen.
#   ryzers run --name flowwam-robotwin /ryzers/demos/demo_flowgen.sh
#   TASK=lift_pot ryzers run --name flowwam-robotwin /ryzers/demos/demo_flowgen.sh
set -euo pipefail
if [ ! -d /opt/RoboTwin ]; then
  echo "ERROR: simulation/robotwin base not found (no /opt/RoboTwin)." >&2
  echo "       Build the chain:  ryzers build robotwin flowwam --name flowwam-robotwin" >&2
  exit 1
fi

FULL_REPO="${FLOWWAM_FULL_REPO:-/repos/flowwam-full}"
FLOWWAM_FULL_COMMIT="${FLOWWAM_FULL_COMMIT:-68abaa2}"
MDIR="${FLOWWAM_MODEL_DIR:-/models/flowwam}"
export TASK="${TASK:-beat_block_hammer}"
export TASK_CONFIG="${TASK_CONFIG:-demo_clean}"
export SEED="${SEED:-0}"
export TAG="${TAG:-robotwin}"
export CKPT="${CKPT:-$MDIR/robotwin/flowwam_robotwin.safetensors}"
export LOCAL_MODEL_PATH="${LOCAL_MODEL_PATH:-$MDIR}"
export OUT_DIR="${OUT_DIR:-/outputs}/flowgen"
export VIDEO_INFERENCE_STEPS="${VIDEO_INFERENCE_STEPS:-25}"
export NUM_VIDEO_FRAMES="${NUM_VIDEO_FRAMES:-9}"
export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.5.1}"
export TORCH_BLAS_PREFER_HIPBLASLT="${TORCH_BLAS_PREFER_HIPBLASLT:-0}"
export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:True}"

# ---- fetch upstream FlowWAM repo (dual-stream pipeline code) + weights (idempotent) ----
if [ ! -d "$FULL_REPO/inference" ]; then
  echo "[setup] cloning upstream FlowWAM -> $FULL_REPO"
  git clone https://github.com/YixiangChen515/FlowWAM.git "$FULL_REPO"
  git -C "$FULL_REPO" checkout "$FLOWWAM_FULL_COMMIT" 2>/dev/null || true
fi
bash /ryzers/scripts/download_checkpoints.sh base
bash /ryzers/scripts/download_checkpoints.sh robotwin
[ -f "$CKPT" ] || { echo "ERROR: missing $CKPT" >&2; exit 1; }

# ---- wire RoboTwin sim assets (needed to render the first-frame scene) ----
SETUP="$(ls /ryzers/scripts/setup_robotwin.sh 2>/dev/null || find / -name setup_robotwin.sh 2>/dev/null | head -1)"
[ -n "$SETUP" ] && bash "$SETUP" || echo "WARN: setup_robotwin.sh not found; assets may be missing" >&2

# server pipeline code (pipeline_loader, dataset_action_robotwin) + full-repo diffsynth + sim.
export PYTHONPATH="$FULL_REPO:$FULL_REPO/inference:/opt/sim:/opt/RoboTwin:${PYTHONPATH:-}"
exec python /ryzers/scripts/flowgen.py
