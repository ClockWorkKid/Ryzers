#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Text-to-World action-controlled (Micro-World-T2W, controlnet on Wan2.1-T2V-1.3B) -> single-column
# MP4 (rule 2.b, no reference). Wraps demos/mw_generate.py --mode t2w (reproduces
# examples/wan2.1/predict_t2w_action_control.py, env-driven).
# Weights: WHICH=t2w scripts/download_checkpoints.sh   (Wan2.1-T2V-1.3B + amd/Micro-World-T2W).
# Knobs: PROMPT ACTION_LIST(json) NEGATIVE_PROMPT SAMPLE_SIZE VIDEO_LENGTH NUM_STEPS GUIDANCE_SCALE
#        SAMPLER SHIFT SEED FPS GPU_MEMORY_MODE TEACACHE TEACACHE_THRESHOLD CFG_SKIP_RATIO
#        TRANSFORMER_PATH(=/models/T2W/transformer) LORA_PATH LORA_WEIGHT
# ACTION_LIST format: [[end_frame, "w s a d shift ctrl _ mouse_y mouse_x"], ..., "space_frames"]
set -euo pipefail
cd "${MW_REPO:-/repos/micro-world}"
exec python "${MW_GEN:-/ryzers/demos/mw_generate.py}" --mode t2w
