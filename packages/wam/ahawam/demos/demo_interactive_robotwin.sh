#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Interactive RoboTwin 2.0 demo (chunk-replay) driven by the AHA-WAM policy, served over
# HTTP/MJPEG. Requires the simulation/robotwin base image (the harness + SAPIEN stack):
#   ryzers build robotwin ahawam
#   ryzers run /ryzers/demos/demo_interactive_robotwin.sh
# View at http://localhost:PORT (ssh -L PORT:localhost:PORT <host>). First run downloads the
# RoboTwin sim assets + checkpoint into the mounted caches. Defaults to AHA-WAM-Flash
# (1 diffusion step) for a responsive demo; set CKPT=.../robotwin_ahawam.pt for the base model.
set -euo pipefail
if [ ! -d /opt/sim/sim_robotwin ]; then
  echo "ERROR: simulation/robotwin base not found (no /opt/sim/sim_robotwin)." >&2
  echo "       Build the chain:  ryzers build robotwin ahawam" >&2
  exit 1
fi
REL=/models/ahawam_release
export TASK="${TASK:-click_bell}"
export TASK_CONFIG="${TASK_CONFIG:-demo_clean}"
export SEED="${SEED:-100000}"
export PORT="${PORT:-8082}"
export OUT_DIR="${OUT_DIR:-/outputs}"
export CKPT="${CKPT:-$REL/robotwin_ahawam-flash.pt}"
export DATASET_STATS="${DATASET_STATS:-$REL/dataset_stats.json}"
export MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
export NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-1}"
export CHUNKS_PER_VIDEO_PREFILL="${CHUNKS_PER_VIDEO_PREFILL:-2}"
export POLICY_FACTORY="ahawam_robotwin_policy:build_policy"
export PYTHONPATH="/opt/ahawam-adapters:/repos/ahawam/src:/repos/ahawam:/repos/ahawam/experiments/robotwin:/opt/sim:/opt/RoboTwin:${PYTHONPATH:-}"

# Fetch the checkpoint if missing (idempotent; resumes from cache).
case "$CKPT" in
  *-flash.pt) bash /ryzers/scripts/download_checkpoints.sh flash ;;
  *)          bash /ryzers/scripts/download_checkpoints.sh robotwin ;;
esac
[ -f "$CKPT" ] || { echo "missing $CKPT -> run scripts/download_checkpoints.sh" >&2; exit 1; }

# Fetch RoboTwin assets + wire them into /opt/RoboTwin (idempotent; sim base script).
bash /ryzers/scripts/setup_robotwin.sh
exec python -m sim_robotwin.interactive_server
