#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Interactive SimplerEnv demo (chunk-replay) driven by the VLA-JEPA policy, served over
# HTTP/MJPEG by the simulation/simplerenv harness. Requires the sim base image:
#   ryzers build simplerenv vlajepa
#   ryzers run /ryzers/demos/demo_interactive_simplerenv.sh
# View at http://localhost:PORT (ssh -L PORT:localhost:PORT <host>). First run downloads the
# VLA-JEPA SimplerEnv checkpoint + Qwen3-VL-2B + V-JEPA2 and the ManiSkill3 Bridge/WidowX assets.
set -uo pipefail
if [ ! -d /opt/sim/sim_simplerenv ]; then
  echo "ERROR: SimplerEnv sim base not found (no /opt/sim/sim_simplerenv)." >&2
  echo "       Build the chain: ryzers build simplerenv vlajepa" >&2
  exit 1
fi

# SAPIEN offscreen Vulkan (RADV) on gfx1151; RENDER_BACKEND auto-detected when empty. Assets
# under the persistent HF-cache mount so they are reused across runs.
export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.5.1}"
export SAPIEN_HEADLESS="${SAPIEN_HEADLESS:-1}"
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/radeon_icd.json}"
export MS_ASSET_DIR="${MS_ASSET_DIR:-/root/.cache/huggingface/maniskill_assets}"

# VLA-JEPA SimplerEnv checkpoint + bridge un-normalization key (see demo_closedloop_simplerenv.sh).
if [ -z "${CKPT_REL:-}" ] || [[ "${CKPT_REL}" != SimplerEnv/* ]]; then
  export CKPT_REL="SimplerEnv/checkpoints/VLA-JEPA-SimplerEnv.pt"
fi
export UNNORM_KEY="${UNNORM_KEY:-oxe_bridge}"
# Interactive demo replays a short chunk per plan (smoother than per-step ensembling).
export REPLAN_STEPS="${REPLAN_STEPS:-5}"
export POLICY_FACTORY="${POLICY_FACTORY:-vlajepa_simplerenv_policy:build_policy}"
export TASK="${TASK:-widowx_put_eggplant_in_basket}"
export PORT="${PORT:-8084}"
export OUT_DIR="${OUT_DIR:-/outputs}"
export PYTHONPATH="/opt/vlajepa-adapters:/ryzers:/repos/VLA-JEPA:/opt/sim:/opt/SimplerEnv:${PYTHONPATH:-}"

bash /ryzers/scripts/setup_simplerenv.sh || true

exec python -u -m sim_simplerenv.interactive_server 2>&1 \
  | grep -viE 'svulkan2|Failed to initialize denoiser|cudaError|No initial pose|Mimic targets'
