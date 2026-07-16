#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# DROID language-conditioned video generation (no server / no sim / no robot): the WAN 14B
# planner "dreams" future frames from real DROID context clips + a text prompt. Writes
# gen/context mp4s to $OUT_DIR/droid_generation. First run needs the DROID checkpoint + the
# frozen Wan2.1-I2V-14B-480P base (download_checkpoints.sh droid + the Wan base).
#   VERA_WAN14B_CKPT_ROOT=/models/wan2.1-i2v-14b-480p \
#   ryzers run /ryzers/demos/demo_videogen.sh
# Optional: MODE={scene1|scene2|both}  TEXT="custom prompt"  SEED=11
set -euo pipefail

: "${VERA_DROID_CKPT_DIR:=/models/vera-ckpts/wan-droid-14b}"
: "${VERA_DROID_CLIPS_DIR:=/models/vera-ckpts/droid-demo-clips}"
: "${VERA_WAN14B_CKPT_ROOT:?set VERA_WAN14B_CKPT_ROOT to the frozen Wan2.1-I2V-14B-480P dir}"
export VERA_DROID_CKPT_DIR VERA_DROID_CLIPS_DIR VERA_WAN14B_CKPT_ROOT

exec python /ryzers/scripts/videogen_droid.py
