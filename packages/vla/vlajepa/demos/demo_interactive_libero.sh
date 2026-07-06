#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Interactive LIBERO demo (chunk-replay) driven by the VLA-JEPA policy, served over
# HTTP/MJPEG by the simulation/libero harness. Requires the sim base image:
#   ryzers build libero vlajepa
#   ryzers run /ryzers/demos/demo_interactive_libero.sh
# View at http://localhost:PORT (ssh -L PORT:localhost:PORT). First run downloads the
# VLA-JEPA checkpoint + Qwen3-VL-2B + V-JEPA2 into the mounted cache.
set -euo pipefail
if [ ! -d /opt/sim/sim_libero ]; then
  echo "ERROR: simulation/libero base not found (no /opt/sim/sim_libero)." >&2
  echo "       Build the chain: ryzers build libero vlajepa" >&2
  exit 1
fi
export SUITE="${SUITE:-libero_object}"
export TASK_ID="${TASK_ID:-0}"
export SEED="${SEED:-1000}"
export PORT="${PORT:-8080}"
export OUT_DIR="${OUT_DIR:-/outputs}"
export POLICY_FACTORY="vlajepa_libero_policy:build_policy"
export PYTHONPATH="/opt/vlajepa-adapters:/ryzers:/repos/VLA-JEPA:/opt/sim:/opt/LIBERO:${PYTHONPATH:-}"
exec python -m sim_libero.interactive_server
