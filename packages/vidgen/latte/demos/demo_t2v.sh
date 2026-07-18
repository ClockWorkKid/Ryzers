#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Text-to-video / text-to-image demo entrypoint with all upstream generation knobs.
# Thin env-var wrapper over demos/demo_t2v.py (built on the upstream LattePipeline).
# Weights: run `WHICH=t2v scripts/download_checkpoints.sh` first -> $LATTE1_DIR (maxin-cn/Latte-1).
#
# Knobs (override from the host: VAR=... ryzers run --name latte /ryzers/demos/demo_t2v.sh):
#   PROMPT        text prompt (single); default: a demo prompt
#   STEPS         diffusion sampling steps           (default 50)
#   RESOLUTION    HxW, each divisible by 8           (default 512x512)
#   FRAMES        video_length; 1 => text-to-image   (default 16)
#   GUIDANCE      classifier-free guidance scale     (default 7.5)
#   METHOD        sampler (DDIM|DDPM|PNDM|EulerDiscrete|DPMSolverMultistep|...)  (default DDIM)
#   SEED          reproducible seed                  (default 0)
#   FPS           output mp4 frame rate              (default 8)
#   TEMPORAL_VAE  1|0 SVD temporal VAE decoder        (default: on for video, off for image)
#   TEMPORAL_ATTN 1|0 temporal attention blocks       (default 1)
#   FP16          1|0 compute precision               (default 1 = fp16)
set -euo pipefail

LATTE_REPO="${LATTE_REPO:-/repos/latte}"
LATTE1_DIR="${LATTE1_DIR:-/models/Latte-1}"
OUT="${OUT_DIR:-/outputs}/t2v_demo"

args=(--model "$LATTE1_DIR" --out "$OUT")
[ -n "${PROMPT:-}" ]     && args+=(--prompt "$PROMPT")
[ -n "${STEPS:-}" ]      && args+=(--steps "$STEPS")
[ -n "${RESOLUTION:-}" ] && args+=(--resolution "$RESOLUTION")
[ -n "${FRAMES:-}" ]     && args+=(--frames "$FRAMES")
[ -n "${GUIDANCE:-}" ]   && args+=(--guidance "$GUIDANCE")
[ -n "${METHOD:-}" ]     && args+=(--method "$METHOD")
[ -n "${SEED:-}" ]       && args+=(--seed "$SEED")
[ -n "${FPS:-}" ]        && args+=(--fps "$FPS")
[ -n "${NUM_PER_PROMPT:-}" ] && args+=(--num-per-prompt "$NUM_PER_PROMPT")
case "${TEMPORAL_VAE:-}" in 1|true|True) args+=(--temporal-vae);; 0|false|False) args+=(--no-temporal-vae);; esac
case "${TEMPORAL_ATTN:-1}" in 0|false|False) args+=(--no-temporal-attn);; esac
case "${FP16:-1}" in 0|false|False) args+=(--fp32);; esac

cd "$LATTE_REPO"
LATTE_REPO="$LATTE_REPO" python "${DEMO_PY:-/ryzers/demos/demo_t2v.py}" "${args[@]}"
