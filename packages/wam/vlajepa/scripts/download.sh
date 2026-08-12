#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Pre-fetch all base assets for VLA-JEPA into the persistent HF cache: the full
# VLA-JEPA checkpoint repo (LIBERO + SimplerEnv sub-checkpoints) plus the Qwen3-VL
# base VLM and the V-JEPA2 encoder. Set HF_TOKEN for gated repos.
#
#   HF_TOKEN=hf_xxx ryzers run /ryzers/download.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HF_COMMON:-$HERE/_hf_common.sh}"

hf_prefetch "${MODEL_REPO:-ginwind/VLA-JEPA}"
hf_prefetch "${BASE_VLM:-Qwen/Qwen3-VL-2B-Instruct}"
hf_prefetch "${BASE_ENCODER:-facebook/vjepa2-vitl-fpc64-256}"
echo "PASS: assets cached under ${HF_HOME:-/root/.cache/huggingface}"
