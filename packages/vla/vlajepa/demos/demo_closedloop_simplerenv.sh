#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Closed-loop SimplerEnv (WidowX / BridgeData v2) rollouts driven by VLA-JEPA. Steps the
# ManiSkill3 real2sim digital twin (CPU PhysX + SAPIEN offscreen Vulkan on gfx1151) and writes
# per-task results + rollout MP4s to /outputs/closedloop/<TAG>/ + an aggregate summary.
# Requires the simulation/simplerenv base image:
#   ryzers build simulation/simplerenv vlajepa
#   ryzers run /ryzers/demos/demo_closedloop_simplerenv.sh
#   TASKS=widowx_put_eggplant_in_basket NUM_TRIALS=10 ryzers run /ryzers/demos/demo_closedloop_simplerenv.sh
# First run downloads the VLA-JEPA SimplerEnv checkpoint + Qwen3-VL-2B + V-JEPA2 and the
# ManiSkill3 Bridge/WidowX scene assets into the mounted caches.
set -uo pipefail
if [ ! -d /opt/sim/sim_simplerenv ]; then
  echo "ERROR: SimplerEnv sim base not found (no /opt/sim/sim_simplerenv)." >&2
  echo "       Build the chain: ryzers build simulation/simplerenv vlajepa" >&2
  exit 1
fi

# SAPIEN offscreen Vulkan (RADV) on gfx1151. RENDER_BACKEND is auto-detected (PCI addr) by
# sim_simplerenv when left empty. Assets go under the persistent HF-cache mount so they are
# reused across runs without adding a volume mapping to this policy layer's config.
export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.5.1}"
export SAPIEN_HEADLESS="${SAPIEN_HEADLESS:-1}"
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/radeon_icd.json}"
export MS_ASSET_DIR="${MS_ASSET_DIR:-/root/.cache/huggingface/maniskill_assets}"

# VLA-JEPA SimplerEnv checkpoint + bridge un-normalization key. The shared vlajepa config
# injects the LIBERO checkpoint as a NON-empty default, so force the SimplerEnv checkpoint
# unless the host explicitly points CKPT_REL at a SimplerEnv one. model_smoke reads CKPT_REL
# at import time, so it must be set here (before python starts), not inside the adapter.
if [ -z "${CKPT_REL:-}" ] || [[ "${CKPT_REL}" != SimplerEnv/* ]]; then
  export CKPT_REL="SimplerEnv/checkpoints/VLA-JEPA-SimplerEnv.pt"
fi
export UNNORM_KEY="${UNNORM_KEY:-oxe_bridge}"
export POLICY_FACTORY="${POLICY_FACTORY:-vlajepa_simplerenv_policy:build_policy}"
export TASKS="${TASKS:-}"
export NUM_TRIALS="${NUM_TRIALS:-5}"
export SEED="${SEED:-0}"
export OUT_DIR="${OUT_DIR:-/outputs}"
export PYTHONPATH="/opt/vlajepa-adapters:/ryzers:/repos/VLA-JEPA:/opt/sim:/opt/SimplerEnv:${PYTHONPATH:-}"

bash /ryzers/scripts/setup_simplerenv.sh || true

# Filter SAPIEN/Vulkan offscreen log noise (as the sim sanity + robotwin demos do).
exec python -u /ryzers/closedloop_simplerenv.py 2>&1 \
  | grep -viE 'svulkan2|Failed to initialize denoiser|cudaError|No initial pose|Mimic targets'
