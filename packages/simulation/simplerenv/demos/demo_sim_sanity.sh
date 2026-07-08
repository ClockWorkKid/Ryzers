#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Headless SimplerEnv sanity: RandomPolicy rollout on one task -> MP4 under /sim_outputs.
# Validates SAPIEN offscreen Vulkan rendering + ManiSkill2 CPU physics on gfx1151.
#   ryzers build simulation/simplerenv
#   TASK=google_robot_pick_coke_can STEPS=40 ryzers run /ryzers/demos/demo_sim_sanity.sh
set -uo pipefail
if [ ! -d /opt/sim/sim_simplerenv ]; then
  echo "ERROR: SimplerEnv sim base not found (no /opt/sim/sim_simplerenv)." >&2
  exit 1
fi
bash /ryzers/scripts/setup_simplerenv.sh || true
export TASK="${TASK:-google_robot_pick_coke_can}"
export STEPS="${STEPS:-40}"
export PYTHONPATH="/opt/sim:/opt/SimplerEnv:${PYTHONPATH:-}"
# Filter SAPIEN/Vulkan offscreen log noise (as robotwin policy demos do).
exec python -u -m sim_simplerenv.sanity 2>&1 | grep -viE 'svulkan2|Failed to initialize denoiser|cudaError'
