#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Real-time LIBERO demo driven by the VLA-JEPA policy: the sim steps at wall-clock RT_HZ
# and the robot HOLDs (pauses, shown as THINKING) while VLA-JEPA plans the next action
# chunk, so planner latency is visible live. Requires the simulation/libero base image
# (the harness + LIBERO stack):
#   ryzers build libero vlajepa
#   ryzers run /ryzers/demos/demo_interactive_libero_rt.sh
# View at http://localhost:PORT (ssh -L PORT:localhost:PORT <host>). First run downloads
# the VLA-JEPA checkpoint + Qwen3-VL-2B + V-JEPA2 into the mounted cache.
set -euo pipefail
if [ ! -d /opt/sim/sim_libero ]; then
  echo "ERROR: simulation/libero base not found (no /opt/sim/sim_libero)." >&2
  echo "       Build the chain:  ryzers build libero vlajepa" >&2
  exit 1
fi
export SUITE="${SUITE:-libero_object}"
export TASK_ID="${TASK_ID:-0}"
export SEED="${SEED:-1000}"
export PORT="${PORT:-8081}"
export RT_HZ="${RT_HZ:-20}"
export OUT_DIR="${OUT_DIR:-/outputs}"
export POLICY_FACTORY="vlajepa_libero_policy:build_policy"
export PYTHONPATH="/opt/vlajepa-adapters:/ryzers:/repos/VLA-JEPA:/opt/sim:/opt/LIBERO:${PYTHONPATH:-}"
exec python -m sim_libero.interactive_server_rt
