#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Pre-fetch ONLY the core VLA-JEPA model assets into the persistent HF cache:
#   - the VLA-JEPA checkpoint repo (config.yaml + dataset_statistics.json + .pt)
#   - the Qwen3-VL base VLM the checkpoint config is repointed to
#   - the V-JEPA2 encoder the checkpoint config is repointed to
# The demos also download these on first run. Set HF_TOKEN for gated repos.
#
#   HF_TOKEN=hf_xxx ryzers run /ryzers/download_smoke.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HF_COMMON:-$HERE/_hf_common.sh}"

# Only the suite CKPT_REL points at (Pretrain/LIBERO/Real-world/SimplerEnv each ship
# a multi-GB .pt); pulling the whole repo would fetch all of them needlessly.
CKPT_REL="${CKPT_REL:-LIBERO/checkpoints/VLA-JEPA-LIBERO.pt}"
SUITE="${CKPT_REL%%/*}"
hf_prefetch "${MODEL_REPO:-ginwind/VLA-JEPA}" --include "${SUITE}/*"
hf_prefetch "${BASE_VLM:-Qwen/Qwen3-VL-2B-Instruct}"
hf_prefetch "${BASE_ENCODER:-facebook/vjepa2-vitl-fpc64-256}"
echo "PASS: smoke assets cached under ${HF_HOME:-/root/.cache/huggingface}"
