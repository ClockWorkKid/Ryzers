#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Single-scene closed-loop LIBERO rollout driven by the MolmoAct2 policy through the shared
# simulation/libero harness (headless EGL), saving a composed agentview|wrist MP4 to
# /outputs. Requires the simulation/libero base image:
#   ryzers build simulation/libero molmoact2
#   ryzers run /ryzers/demos/demo_closedloop_libero.sh
#   SUITE=libero_goal TASK_ID=2 STEPS=200 ryzers run /ryzers/demos/demo_closedloop_libero.sh
#
# For full per-suite success-rate reproduction, iterate this over tasks (mirror
# wam/fastwam's demo_closedloop_libero.sh aggregation) once the adapter is validated.
# NOTE: shared-base MolmoAct2 path is under validation on strix-halo -- see
# docs/LIBERO_SIMBASE_ADAPTATION.md.
set -euo pipefail
if [ ! -d /opt/sim/sim_libero ]; then
  echo "ERROR: simulation/libero base not found (no /opt/sim/sim_libero)." >&2
  echo "       Build the chain:  ryzers build simulation/libero molmoact2" >&2
  exit 1
fi
export SUITE="${SUITE:-libero_object}"
export TASK_ID="${TASK_ID:-0}"
export SEED="${SEED:-1000}"
export STEPS="${STEPS:-200}"
export OUT_DIR="${OUT_DIR:-/outputs}"
export CKPT="${CKPT:-allenai/MolmoAct2-Think-LIBERO}"
export THINK="${THINK:-1}"
export POLICY_FACTORY="molmoact2_libero_policy:build_policy"
export PYTHONPATH="/opt/molmoact2-adapters:/repos/molmoact2:/opt/sim:/opt/LIBERO:${PYTHONPATH:-}"

exec python -m sim_libero.sanity
