#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Steady-state latency breakdown of one infer_action (text encoder / VAE / world
# prefill / flow-matching plan) and an SDPA attention-backend comparison.
#   ryzers run /ryzers/demos/demo_latency.sh
#   DATASET=robotwin ryzers run /ryzers/demos/demo_latency.sh
set -euo pipefail
DATASET="${DATASET:-libero}"
REL=/models/fastwam_release
export FASTWAM_REPO=/repos/fastwam
export PYTHONPATH="/repos/fastwam/src:/repos/fastwam:${PYTHONPATH:-}"

case "$DATASET" in
  libero)   export CONFIG_NAME=sim_libero   CKPT=$REL/libero_uncond_2cam224.pt ;;
  robotwin) export CONFIG_NAME=sim_robotwin CKPT=$REL/robotwin_uncond_3cam_384.pt ;;
  *) echo "DATASET must be libero|robotwin" >&2; exit 2 ;;
esac
export CKPT
[ -f "$CKPT" ] || { echo "missing $CKPT -> run scripts/download_checkpoints.sh $DATASET" >&2; exit 1; }

exec python /ryzers/scripts/planning_bench.py
