#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Two-phase latency breakdown for AHA-WAM: slow video-context prefill (planner) vs
# fast action-chunk executor (control Hz), plus an SDPA attention-backend comparison.
# This quantifies why the async runtime decouples the branches.
#   ryzers run /ryzers/demos/demo_latency.sh                    # AHA-WAM-Flash (1-step)
#   WHICH=robotwin NUM_STEPS=10 ryzers run /ryzers/demos/demo_latency.sh   # base AHA-WAM
set -euo pipefail
REL=/models/ahawam_release
WHICH="${WHICH:-flash}"
case "$WHICH" in
  flash)    CKPT="${CKPT:-$REL/robotwin_ahawam-flash.pt}"; DEF_STEPS=1 ;;
  robotwin) CKPT="${CKPT:-$REL/robotwin_ahawam.pt}";       DEF_STEPS=10 ;;
  *) echo "WHICH must be flash|robotwin" >&2; exit 2 ;;
esac
export AHAWAM_REPO=/repos/ahawam
export CONFIG_NAME="${CONFIG_NAME:-sim_robotwin}"
export CKPT
export NUM_STEPS="${NUM_STEPS:-$DEF_STEPS}"
export PYTHONPATH="/repos/ahawam/src:/repos/ahawam:${PYTHONPATH:-}"

bash /ryzers/scripts/download_checkpoints.sh "$WHICH"
[ -f "$CKPT" ] || { echo "missing $CKPT -> run scripts/download_checkpoints.sh $WHICH" >&2; exit 1; }

exec python /ryzers/scripts/planning_bench.py
