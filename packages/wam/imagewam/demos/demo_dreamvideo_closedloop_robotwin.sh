#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Continuous closed-loop "dream vs sim" VIDEO on RoboTwin 2.0 (SAPIEN, offscreen Vulkan). The
# policy runs the rollout; we capture the ACTUAL sim frame every step (3-cam compact layout,
# left column) and the model's re-dreamed future (infer_video_flux2) at each replan (right
# column), stitched into one smooth side-by-side MP4 (cl_dreamvid_robotwin_<task>.mp4).
# Requires the sim chain:  ryzers build robotwin imagewam
#   ryzers run /ryzers/demos/demo_dreamvideo_closedloop_robotwin.sh
#   TASK=beat_block_hammer TASK_CONFIG=demo_clean ryzers run /ryzers/demos/demo_dreamvideo_closedloop_robotwin.sh
set -euo pipefail

if [ ! -d /opt/RoboTwin ]; then
  echo "ERROR: simulation/robotwin base not found (no /opt/RoboTwin)." >&2
  echo "       Build the chain:  ryzers build robotwin imagewam" >&2
  exit 1
fi

REPO="${IMAGEWAM_REPO:-/repos/imagewam}"
FLUX2_SRC="${FLUX2_SRC:-/repos/flux2}"
export PYTHONPATH="${REPO}/src:${FLUX2_SRC}/src:${FLUX2_SRC}:${REPO}/experiments/robotwin:/opt/RoboTwin:${PYTHONPATH:-}"

bash /ryzers/scripts/download_checkpoints.sh robotwin "${FLUX2_VARIANT:-4b}"

# Fetch RoboTwin assets + wire them into /opt/RoboTwin (idempotent; provided by the sim base).
bash /ryzers/scripts/setup_robotwin.sh

exec python /ryzers/scripts/dream_rollout_video_robotwin.py
