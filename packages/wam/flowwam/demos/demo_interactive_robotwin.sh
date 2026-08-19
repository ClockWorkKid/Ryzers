#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Interactive RoboTwin demo (chunk-replay; FlowWAM dual-stream flow-action policy over
# HTTP/MJPEG). The interactive server (from the robotwin sim base) streams the live 4-view to
# the browser; the FlowWAM policy is wired in via POLICY_FACTORY, which launches the upstream
# flow_action_server (same model + checkpoint as the closed loop) and drives it through the
# robotwin_policy websocket client. Requires the simulation/robotwin base:
#   ryzers build robotwin flowwam --name flowwam-robotwin
#   ryzers run --name flowwam-robotwin /ryzers/demos/demo_interactive_robotwin.sh
# View at http://localhost:PORT (ssh -L PORT:localhost:PORT <host>).
set -euo pipefail
if [ ! -d /opt/sim/sim_robotwin ]; then
  echo "ERROR: simulation/robotwin base not found (no /opt/sim/sim_robotwin)." >&2
  echo "       Build the chain:  ryzers build robotwin flowwam --name flowwam-robotwin" >&2
  exit 1
fi

FULL_REPO="${FLOWWAM_FULL_REPO:-/repos/flowwam-full}"
FLOWWAM_FULL_COMMIT="${FLOWWAM_FULL_COMMIT:-68abaa2}"
MDIR="${FLOWWAM_MODEL_DIR:-/models/flowwam}"
export TASK="${TASK:-beat_block_hammer}"
export TASK_CONFIG="${TASK_CONFIG:-demo_clean}"
export SEED="${SEED:-100000}"
export PORT="${PORT:-8082}"                          # interactive HTTP/MJPEG port
export FLOW_SERVER_PORT="${FLOW_SERVER_PORT:-8000}"  # flow_action_server ws port (internal)
export OUT_DIR="${OUT_DIR:-/sim_outputs}"
export CKPT="${CKPT:-$MDIR/robotwin/flowwam_robotwin.safetensors}"
export ACTION_NORM_PATH="${ACTION_NORM_PATH:-$MDIR/robotwin/flowwam_robotwin_action_norm_stats.npz}"
export LOCAL_MODEL_PATH="${LOCAL_MODEL_PATH:-$MDIR}"
export EXECUTE_WINDOW="${EXECUTE_WINDOW:-25}"
export VIDEO_INFERENCE_STEPS="${VIDEO_INFERENCE_STEPS:-25}"
export ACTION_INFERENCE_STEPS="${ACTION_INFERENCE_STEPS:-50}"
export POLICY_FACTORY="flowwam_robotwin_policy:build_policy"
export PYTHONPATH="/opt/flowwam-adapters:$FULL_REPO/inference/robotwin_policy:/opt/sim:/opt/RoboTwin:${PYTHONPATH:-}"

# ---- fetch upstream FlowWAM repo (server + robotwin_policy client) + weights (idempotent) ----
if [ ! -d "$FULL_REPO/inference" ]; then
  echo "[setup] cloning upstream FlowWAM (action policy) -> $FULL_REPO"
  git clone https://github.com/YixiangChen515/FlowWAM.git "$FULL_REPO"
  git -C "$FULL_REPO" checkout "$FLOWWAM_FULL_COMMIT" 2>/dev/null || true
fi
bash /ryzers/scripts/download_checkpoints.sh base
bash /ryzers/scripts/download_checkpoints.sh robotwin
[ -f "$CKPT" ] || { echo "ERROR: missing $CKPT" >&2; exit 1; }

# ---- wire RoboTwin sim assets (embodiments/objects/task_config) into /opt/RoboTwin ----
SETUP="$(ls /ryzers/scripts/setup_robotwin.sh 2>/dev/null || find / -name setup_robotwin.sh 2>/dev/null | head -1)"
[ -n "$SETUP" ] && bash "$SETUP" || echo "WARN: setup_robotwin.sh not found; assets may be missing" >&2

exec python -m sim_robotwin.interactive_server
