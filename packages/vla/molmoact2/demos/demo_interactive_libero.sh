#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Interactive LIBERO demo (chunk-replay) driven by the MolmoAct2 policy through the shared
# simulation/libero harness. Requires the simulation/libero base image (harness + LIBERO):
#   ryzers build simulation/libero molmoact2
#   ryzers run /ryzers/demos/demo_interactive_libero.sh
# View at http://localhost:PORT (ssh -L PORT:localhost:PORT <host>).
#
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
export PORT="${PORT:-8080}"
export OUT_DIR="${OUT_DIR:-/outputs}"
export CKPT="${CKPT:-allenai/MolmoAct2-Think-LIBERO}"
export THINK="${THINK:-1}"
export POLICY_FACTORY="molmoact2_libero_policy:build_policy"
export PYTHONPATH="/opt/molmoact2-adapters:/repos/molmoact2:/opt/sim:/opt/LIBERO:${PYTHONPATH:-}"

exec python -m sim_libero.interactive_server
