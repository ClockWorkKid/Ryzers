#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Text-to-video (Wan2.1-T2V-1.3B baseline, no action control) -> single-column MP4 (rule 2.b).
# Wraps demos/mw_generate.py --mode t2v (reproduces examples/wan2.1/predict_t2v.py, env-driven).
# Weights: WHICH=t2v scripts/download_checkpoints.sh   (Wan-AI/Wan2.1-T2V-1.3B).
# Knobs (override from host: VAR=... ryzers run --name micro-world /ryzers/demos/demo_t2v.sh):
#   PROMPT NEGATIVE_PROMPT SAMPLE_SIZE(=H,W) VIDEO_LENGTH NUM_STEPS GUIDANCE_SCALE SAMPLER SHIFT
#   SEED FPS GPU_MEMORY_MODE TEACACHE TEACACHE_THRESHOLD CFG_SKIP_RATIO
set -euo pipefail
cd "${MW_REPO:-/repos/micro-world}"
exec python "${MW_GEN:-/ryzers/demos/mw_generate.py}" --mode t2v
