#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Real-time LIBERO-Plus demo driven by the VLA-JEPA policy: the sim steps at wall-clock
# RT_HZ and the robot HOLDs (shown as THINKING) while VLA-JEPA plans the next action chunk,
# so planner latency is visible live on a perturbed scene. Requires the simulation/libero-plus
# base image (the harness + LIBERO-Plus stack):
#   ryzers build libero-plus vlajepa
#   ryzers run /ryzers/demos/demo_interactive_liberoplus_rt.sh
# View at http://localhost:PORT (ssh -L PORT:localhost:PORT <host>). First run downloads
# the VLA-JEPA checkpoint + Qwen3-VL-2B + V-JEPA2 into the mounted cache.
set -euo pipefail
if [ ! -d /opt/sim/sim_liberoplus ]; then
  echo "ERROR: simulation/libero-plus base not found (no /opt/sim/sim_liberoplus)." >&2
  echo "       Build the chain:  ryzers build libero-plus vlajepa" >&2
  exit 1
fi
export SUITE="${SUITE:-libero_object}"
export TASK_ID="${TASK_ID:-0}"
export SEED="${SEED:-1000}"
export PORT="${PORT:-8081}"
export RT_HZ="${RT_HZ:-20}"
export OUT_DIR="${OUT_DIR:-/outputs}"
export POLICY_FACTORY="vlajepa_liberoplus_policy:build_policy"
export PYTHONPATH="/opt/vlajepa-adapters:/ryzers:/repos/VLA-JEPA:/opt/sim:/opt/LIBERO-plus:${PYTHONPATH:-}"
exec python -m sim_liberoplus.interactive_server_rt
