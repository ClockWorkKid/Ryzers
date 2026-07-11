#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Interactive LIBERO demo (chunk-replay) driven by the MolmoAct2 policy through the shared
# simulation/libero harness, served over HTTP/MJPEG. Requires the simulation/libero base:
#   ryzers build simulation/libero molmoact2
#   ryzers run /ryzers/demo_interactive_libero.sh
# View at http://localhost:PORT (remote box: ssh -L PORT:localhost:PORT <host>). Type an
# instruction to reset the scene and run it; use the dropdown to switch LIBERO tasks.
#
# The MolmoAct2 lerobot policy runs in the isolated /opt/libero-venv behind a localhost
# policy server; the harness (base venv) drives it via adapters/molmoact2_libero_policy.py.
# Fast path by default (THINK=0, NUM_STEPS=4); pass THINK=1 for full depth reasoning.
set -euo pipefail
if [ ! -d /opt/sim/sim_libero ]; then
  echo "ERROR: simulation/libero base not found (no /opt/sim/sim_libero)." >&2
  echo "       Build the chain:  ryzers build simulation/libero molmoact2" >&2
  exit 1
fi
export SUITE="${SUITE:-libero_object}"
export TASK_ID="${TASK_ID:-0}"
export SEED="${SEED:-1000}"
export PORT="${PORT:-8080}"
export OUT_DIR="${OUT_DIR:-/outputs}"
export CKPT="${CKPT:-allenai/MolmoAct2-Think-LIBERO}"
export THINK="${THINK:-0}"
export NUM_STEPS="${NUM_STEPS:-4}"
export MM2_SERVER_PORT="${MM2_SERVER_PORT:-8790}"
export POLICY_FACTORY="molmoact2_libero_policy:build_policy"
export PYTHONPATH="/opt/molmoact2-adapters:/opt/sim:/opt/LIBERO:${PYTHONPATH:-}"

source /ryzers/start_policy_server.sh
echo "MolmoAct2 interactive | suite=$SUITE task_id=$TASK_ID seed=$SEED think=$THINK num_steps=$NUM_STEPS port=$PORT"
echo "Open http://localhost:$PORT in your browser (remote box: ssh -L $PORT:localhost:$PORT <host>)"
export SIM_HARNESS_MODULE="sim_libero.interactive_server"
exec python /ryzers/molmoact2_run_harness.py
