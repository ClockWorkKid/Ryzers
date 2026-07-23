#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# P8 per-module latency / computation profile of FLUX.2 ImageWAM: instruments the ACTION path
# (infer_action_flux2, deployment/closed-loop) and the DREAM path (infer_video_flux2, image
# world-model / viz), reporting the fixed-vs-per-step split, param distribution, and the
# MEASURED text-embedding cache win (instruction is fixed per episode -> encode Qwen3 once).
# Inputs are synthetic tensors of the correct shape (latency is shape- not content-dependent),
# so NO dataset or simulator is needed -- runs on the plain imagewam image.
# Writes p8_latency.json + p8_latency.png to /outputs. Usage:
#   ryzers run --name imagewam /ryzers/demos/demo_latency.sh
#   NUM_STEPS=20 ITERS=10 WARMUP=3 ryzers run --name imagewam /ryzers/demos/demo_latency.sh
set -euo pipefail

# Ensure weights are present (idempotent; resumes from cache). Gated FLUX.2 base/AE are
# skipped with a warning if HF_TOKEN is unset -- the profiler only needs shapes, but load
# uses the real checkpoint so keep the download here for a faithful measurement.
bash /ryzers/scripts/download_checkpoints.sh "${SUITE:-libero}" "${FLUX2_VARIANT:-4b}"

exec python /ryzers/scripts/profile_modules.py
