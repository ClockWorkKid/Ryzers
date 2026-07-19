#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Image-to-video (Wan2.1-I2V-14B-480P baseline, no action control) -> two-column MP4:
# reference image (left) | generated video (right) (rule 2.b). Wraps demos/mw_generate.py --mode i2v
# (reproduces examples/wan2.1/predict_i2v.py, env-driven).
# Weights: WHICH=i2v scripts/download_checkpoints.sh   (Wan-AI/Wan2.1-I2V-14B-480P).
# Knobs: REF_IMAGE(=asset/cliff.jpg) PROMPT NEGATIVE_PROMPT SAMPLE_SIZE VIDEO_LENGTH NUM_STEPS
#        GUIDANCE_SCALE SAMPLER SHIFT SEED FPS GPU_MEMORY_MODE TEACACHE TEACACHE_THRESHOLD CFG_SKIP_RATIO
# NOTE: 14B model — if VRAM is tight, set GPU_MEMORY_MODE=model_cpu_offload.
set -euo pipefail
cd "${MW_REPO:-/repos/micro-world}"
exec python "${MW_GEN:-/ryzers/demos/mw_generate.py}" --mode i2v
