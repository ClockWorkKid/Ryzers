#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Continuous closed-loop "dream vs sim" VIDEO on LIBERO (MuJoCo, headless). The policy runs the
# rollout in the simulator; we capture the ACTUAL sim frame (agentview | wrist) at EVERY step
# (left column = continuous simulator output) and the model's re-dreamed future
# (infer_video_flux2) at each replan (right column), stitched into one smooth side-by-side MP4
# (cl_dreamvid_<suite>_task<id>.mp4). Improves on the sparse still-panel dream_rollout_libero.py.
# Requires the sim chain:  ryzers build libero imagewam
#   ryzers run /ryzers/demos/demo_dreamvideo_closedloop_libero.sh
#   SUITE=libero_object TASK_ID=0 MAX_STEPS=260 ryzers run /ryzers/demos/demo_dreamvideo_closedloop_libero.sh
set -euo pipefail

if [ ! -d /opt/LIBERO ]; then
  echo "ERROR: simulation/libero base not found (no /opt/LIBERO)." >&2
  echo "       Build the chain:  ryzers build libero imagewam" >&2
  exit 1
fi

REPO="${IMAGEWAM_REPO:-/repos/imagewam}"
FLUX2_SRC="${FLUX2_SRC:-/repos/flux2}"
export PYTHONPATH="${REPO}/src:${FLUX2_SRC}/src:${FLUX2_SRC}:${REPO}/experiments/libero:/opt/LIBERO:${PYTHONPATH:-}"

bash /ryzers/scripts/download_checkpoints.sh libero "${FLUX2_VARIANT:-4b}"

exec python /ryzers/scripts/dream_rollout_video_libero.py
