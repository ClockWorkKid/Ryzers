#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Closed-loop LIBERO rollouts in MuJoCo (headless EGL) driven by VLA-JEPA. Writes
# per-task results + rollout MP4s to /outputs/closedloop/<TAG>/<SUITE>/ and an
# aggregate success_summary.json. Requires the simulation/libero base image:
#   ryzers build libero vlajepa
#   ryzers run /ryzers/demos/demo_closedloop_libero.sh
#   SUITE=libero_goal NUM_TASKS=2 NUM_TRIALS=10 ryzers run /ryzers/demos/demo_closedloop_libero.sh
# First run downloads the VLA-JEPA checkpoint + Qwen3-VL-2B + V-JEPA2 into the mounted cache.
set -uo pipefail
if [ ! -d /opt/sim/sim_libero ]; then
  echo "ERROR: simulation/libero base not found (no /opt/sim/sim_libero)." >&2
  echo "       Build the chain: ryzers build libero vlajepa" >&2
  exit 1
fi
export SUITE="${SUITE:-libero_object}"
export NUM_TASKS="${NUM_TASKS:-3}"
export NUM_TRIALS="${NUM_TRIALS:-5}"
export SEED="${SEED:-1000}"
export OUT_DIR="${OUT_DIR:-/outputs}"
export POLICY_FACTORY="${POLICY_FACTORY:-vlajepa_libero_policy:build_policy}"
export PYTHONPATH="/opt/vlajepa-adapters:/ryzers:/repos/VLA-JEPA:/opt/sim:/opt/LIBERO:${PYTHONPATH:-}"
exec python /ryzers/closedloop_libero.py
