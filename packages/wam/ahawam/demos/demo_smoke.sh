#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Full-model smoke: load the released AHA-WAM RoboTwin 2.0 checkpoint and run one
# end-to-end action prediction on ROCm (two-phase: video-context prefill + action
# chunk). First run downloads the Wan2.2 base weights + checkpoint into the mounted
# cache. Usage:
#   ryzers run /ryzers/demos/demo_smoke.sh
#   CKPT=/models/ahawam_release/robotwin_ahawam-flash.pt ryzers run /ryzers/demos/demo_smoke.sh
set -euo pipefail

# Ensure the checkpoint is present (idempotent; resumes from cache).
bash /ryzers/scripts/download_checkpoints.sh robotwin

exec python /ryzers/scripts/model_smoke.py
