#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Image-to-World action-controlled (Micro-World-I2W, adaln + lora on Wan2.1-I2V-14B-480P) ->
# two-column MP4: reference image (left) | generated video (right) (rule 2.b). Wraps
# demos/mw_generate.py --mode i2w (reproduces examples/wan2.1/predict_i2w_action_control.py).
# Weights: WHICH=i2w scripts/download_checkpoints.sh   (Wan2.1-I2V-14B-480P + amd/Micro-World-I2W).
# Knobs: REF_IMAGE(=asset/street_night.jpg) PROMPT ACTION_LIST(json) NEGATIVE_PROMPT SAMPLE_SIZE
#        VIDEO_LENGTH NUM_STEPS GUIDANCE_SCALE SAMPLER SHIFT SEED FPS GPU_MEMORY_MODE TEACACHE
#        TEACACHE_THRESHOLD CFG_SKIP_RATIO TRANSFORMER_PATH(=/models/I2W/transformer) LORA_PATH LORA_WEIGHT
# NOTE: 14B model — if VRAM is tight, set GPU_MEMORY_MODE=model_cpu_offload.
set -euo pipefail
cd "${MW_REPO:-/repos/micro-world}"
exec python "${MW_GEN:-/ryzers/demos/mw_generate.py}" --mode i2w
