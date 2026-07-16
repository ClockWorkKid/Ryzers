#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Fixed-image + prompt -> video rollout: start from a single still frame (or three per-camera
# stills) and a language prompt; the WAN 14B planner dreams the future rollout. No server / sim
# / robot. Writes a side-by-side [held input | generated] mp4 + a generated-only mp4 to
# $OUT_DIR/img2vid. Needs the DROID checkpoint + frozen Wan2.1-I2V-14B-480P base.
#   IMAGE=/inputs/start.png TEXT="a white robot arm picks up the red block" \
#   VERA_WAN14B_CKPT_ROOT=/models/wan2.1-i2v-14b-480p \
#   ryzers run /ryzers/demos/demo_img2vid.sh
# Optional: VIEWS="/inputs/ext.png,/inputs/side.png,/inputs/wrist.png"  NAME=run1  SEED=11
set -euo pipefail

: "${VERA_DROID_CKPT_DIR:=/models/vera-ckpts/wan-droid-14b}"
: "${VERA_WAN14B_CKPT_ROOT:?set VERA_WAN14B_CKPT_ROOT to the frozen Wan2.1-I2V-14B-480P dir}"
: "${TEXT:?set TEXT to a language prompt}"
export VERA_DROID_CKPT_DIR VERA_WAN14B_CKPT_ROOT TEXT

exec python /ryzers/scripts/videogen_image2video.py
