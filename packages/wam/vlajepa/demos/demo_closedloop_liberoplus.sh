#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Closed-loop LIBERO-Plus robustness rollouts in MuJoCo (headless EGL) driven by VLA-JEPA.
# Selects perturbation tasks by CATEGORY/DIFFICULTY, runs 1 trial/task, and writes an
# aggregate robustness_summary.json (overall + per-dimension + per-difficulty) plus rollout
# MP4s to /outputs/liberoplus/<TAG>/<SUITE>/. Requires the simulation/libero-plus base image:
#   ryzers build libero-plus vlajepa
#   ryzers run /ryzers/demos/demo_closedloop_liberoplus.sh
#   CATEGORY="Camera Viewpoints" MAX_TASKS=60 ryzers run /ryzers/demos/demo_closedloop_liberoplus.sh
#   SUITE=libero_goal DIFFICULTY=3 ryzers run /ryzers/demos/demo_closedloop_liberoplus.sh
# First run downloads the VLA-JEPA checkpoint + Qwen3-VL-2B + V-JEPA2 into the mounted cache.
set -uo pipefail
if [ ! -d /opt/sim/sim_liberoplus ]; then
  echo "ERROR: simulation/libero-plus base not found (no /opt/sim/sim_liberoplus)." >&2
  echo "       Build the chain: ryzers build libero-plus vlajepa" >&2
  exit 1
fi
export SUITE="${SUITE:-libero_object}"
export CATEGORY="${CATEGORY:-}"
export DIFFICULTY="${DIFFICULTY:-}"
export MAX_TASKS="${MAX_TASKS:-40}"
export NUM_TRIALS="${NUM_TRIALS:-1}"
export SEED="${SEED:-1000}"
export OUT_DIR="${OUT_DIR:-/outputs}"
export POLICY_FACTORY="${POLICY_FACTORY:-vlajepa_liberoplus_policy:build_policy}"
export PYTHONPATH="/opt/vlajepa-adapters:/ryzers:/repos/VLA-JEPA:/opt/sim:/opt/LIBERO-plus:${PYTHONPATH:-}"
exec python /ryzers/closedloop_liberoplus.py
