#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Single-scene closed-loop LIBERO rollout driven by the MolmoAct2 policy through the shared
# simulation/libero harness (headless EGL), saving a composed agentview|wrist MP4 to
# /outputs. Requires the simulation/libero base image:
#   ryzers build simulation/libero molmoact2
#   ryzers run /ryzers/demo_closedloop_libero.sh
#   SUITE=libero_goal TASK_ID=2 STEPS=400 ryzers run /ryzers/demo_closedloop_libero.sh
#
# The MolmoAct2 lerobot policy (lerobot 0.5.1 / transformers 5.3 / numpy 2) runs in the
# isolated /opt/libero-venv behind a localhost policy server; the harness (numpy 1.26 /
# robosuite 1.4) drives it via adapters/molmoact2_libero_policy.py.
set -euo pipefail
if [ ! -d /opt/sim/sim_libero ]; then
  echo "ERROR: simulation/libero base not found (no /opt/sim/sim_libero)." >&2
  echo "       Build the chain:  ryzers build simulation/libero molmoact2" >&2
  exit 1
fi
export SUITE="${SUITE:-libero_object}"
export TASK_ID="${TASK_ID:-0}"
export SEED="${SEED:-1000}"
export STEPS="${STEPS:-400}"
export OUT_DIR="${OUT_DIR:-/outputs}"
export CKPT="${CKPT:-allenai/MolmoAct2-Think-LIBERO}"
export THINK="${THINK:-1}"
export MM2_SERVER_PORT="${MM2_SERVER_PORT:-8790}"
export POLICY_FACTORY="molmoact2_libero_policy:build_policy"
export PYTHONPATH="/opt/molmoact2-adapters:/opt/sim:/opt/LIBERO:${PYTHONPATH:-}"

source /ryzers/start_policy_server.sh
echo "MolmoAct2 closed-loop | suite=$SUITE task_id=$TASK_ID seed=$SEED think=$THINK steps=$STEPS ckpt=$CKPT"
export SIM_HARNESS_MODULE="sim_libero.sanity"
exec python /ryzers/molmoact2_run_harness.py
