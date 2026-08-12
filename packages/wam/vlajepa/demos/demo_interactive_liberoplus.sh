#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Interactive LIBERO-Plus demo (chunk-replay) driven by the VLA-JEPA policy, served over
# HTTP/MJPEG by the simulation/libero-plus harness. TASK_ID selects a perturbation instance
# (the scene shows the perturbed camera/light/background/etc). Requires the sim base image:
#   ryzers build libero-plus vlajepa
#   ryzers run /ryzers/demos/demo_interactive_liberoplus.sh
# View at http://localhost:PORT (ssh -L PORT:localhost:PORT). First run downloads the
# VLA-JEPA checkpoint + Qwen3-VL-2B + V-JEPA2 into the mounted cache.
set -euo pipefail
if [ ! -d /opt/sim/sim_liberoplus ]; then
  echo "ERROR: simulation/libero-plus base not found (no /opt/sim/sim_liberoplus)." >&2
  echo "       Build the chain: ryzers build libero-plus vlajepa" >&2
  exit 1
fi
export SUITE="${SUITE:-libero_object}"
export TASK_ID="${TASK_ID:-0}"
export SEED="${SEED:-1000}"
export PORT="${PORT:-8080}"
export OUT_DIR="${OUT_DIR:-/outputs}"
export POLICY_FACTORY="vlajepa_liberoplus_policy:build_policy"
export PYTHONPATH="/opt/vlajepa-adapters:/ryzers:/repos/VLA-JEPA:/opt/sim:/opt/LIBERO-plus:${PYTHONPATH:-}"
exec python -m sim_liberoplus.interactive_server
