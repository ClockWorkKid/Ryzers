#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Real-time SimplerEnv demo driven by the VLA-JEPA policy: the sim steps at wall-clock RT_HZ
# and the arm HOLDs (shown as THINKING, zero delta) while VLA-JEPA plans the next chunk, so
# planner latency is visible live. Requires the simulation/simplerenv base image:
#   ryzers build simplerenv vlajepa
#   ryzers run /ryzers/demos/demo_interactive_simplerenv_rt.sh
# View at http://localhost:PORT (ssh -L PORT:localhost:PORT <host>).
set -uo pipefail
if [ ! -d /opt/sim/sim_simplerenv ]; then
  echo "ERROR: SimplerEnv sim base not found (no /opt/sim/sim_simplerenv)." >&2
  echo "       Build the chain: ryzers build simplerenv vlajepa" >&2
  exit 1
fi

export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.5.1}"
export SAPIEN_HEADLESS="${SAPIEN_HEADLESS:-1}"
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/radeon_icd.json}"
export MS_ASSET_DIR="${MS_ASSET_DIR:-/root/.cache/huggingface/maniskill_assets}"

if [ -z "${CKPT_REL:-}" ] || [[ "${CKPT_REL}" != SimplerEnv/* ]]; then
  export CKPT_REL="SimplerEnv/checkpoints/VLA-JEPA-SimplerEnv.pt"
fi
export UNNORM_KEY="${UNNORM_KEY:-oxe_bridge}"
# Chunk-replay so execution can run ahead of the planner and HOLD (THINKING) gaps are visible.
export REPLAN_STEPS="${REPLAN_STEPS:-5}"
export POLICY_FACTORY="${POLICY_FACTORY:-vlajepa_simplerenv_policy:build_policy}"
export TASK="${TASK:-widowx_put_eggplant_in_basket}"
export PORT="${PORT:-8085}"
export RT_HZ="${RT_HZ:-5}"
export OUT_DIR="${OUT_DIR:-/outputs}"
export PYTHONPATH="/opt/vlajepa-adapters:/ryzers:/repos/VLA-JEPA:/opt/sim:/opt/SimplerEnv:${PYTHONPATH:-}"

bash /ryzers/scripts/setup_simplerenv.sh || true

exec python -u -m sim_simplerenv.interactive_server_rt 2>&1 \
  | grep -viE 'svulkan2|Failed to initialize denoiser|cudaError|No initial pose|Mimic targets'
