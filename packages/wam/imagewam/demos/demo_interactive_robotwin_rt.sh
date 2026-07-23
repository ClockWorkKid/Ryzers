#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Real-time interactive RoboTwin demo driven by the FLUX.2 ImageWAM policy, served over
# HTTP/MJPEG. Execution is decoupled from planning so the browser SEES planner latency
# (arms HOLD while the model thinks). Requires the simulation/robotwin base:
#   ryzers build robotwin imagewam
#   ryzers run /ryzers/demos/demo_interactive_robotwin_rt.sh
# View at http://localhost:PORT (ssh -L PORT:localhost:PORT <host>).
set -euo pipefail
if [ ! -d /opt/sim/sim_robotwin ]; then
  echo "ERROR: simulation/robotwin base not found (no /opt/sim/sim_robotwin)." >&2
  echo "       Build the chain:  ryzers build robotwin imagewam" >&2
  exit 1
fi
REPO="${IMAGEWAM_REPO:-/repos/imagewam}"
FLUX2_SRC="${FLUX2_SRC:-/repos/flux2}"
VARIANT="${FLUX2_VARIANT:-4b}"
export TASK="${TASK:-click_bell}"
export TASK_CONFIG="${TASK_CONFIG:-demo_clean}"
export SEED="${SEED:-100000}"
export PORT="${PORT:-8083}"
export OUT_DIR="${OUT_DIR:-/outputs}"
export CKPT="${CKPT:-/models/imagewam_release/robotwin/flux2_klein_${VARIANT}/model.pt}"
export DATASET_STATS="${DATASET_STATS:-/models/imagewam_release/robotwin/flux2_klein_${VARIANT}/dataset_stats.json}"
export FLUX2_SRC FLUX2_VARIANT="${VARIANT}"
export FLUX2_MODEL_PATH="${FLUX2_MODEL_PATH:-/models/flux2/FLUX.2-klein-base-4B/flux-2-klein-base-4b.safetensors}"
export FLUX2_AE_MODEL_PATH="${FLUX2_AE_MODEL_PATH:-/models/flux2/FLUX.2-klein-base-4B/ae.safetensors}"
export FLUX2_QWEN3_MODEL_SPEC="${FLUX2_QWEN3_MODEL_SPEC:-Qwen/Qwen3-4B}"
export REPLAN_STEPS="${REPLAN_STEPS:-16}" NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-10}"
export POLICY_FACTORY="imagewam_robotwin_policy:build_policy"
export PYTHONPATH="/opt/imagewam-adapters:${REPO}:${REPO}/src:${FLUX2_SRC}/src:${FLUX2_SRC}:${REPO}/experiments/robotwin:/opt/sim:/opt/RoboTwin:${PYTHONPATH:-}"

for f in "$CKPT" "$DATASET_STATS" "$FLUX2_MODEL_PATH" "$FLUX2_AE_MODEL_PATH"; do
  [ -f "$f" ] || { echo "missing weight: $f" >&2; exit 1; }
done
bash /ryzers/scripts/setup_robotwin.sh
exec python -m sim_robotwin.interactive_server_rt
