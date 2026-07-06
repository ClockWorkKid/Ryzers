#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Pre-fetch everything the open-loop demo needs: the model assets (via
# download_smoke.sh) plus ONE LIBERO demo HDF5 selected by SUITE/TASK_ID.
#
#   SUITE=libero_object TASK_ID=0 ryzers run /ryzers/download_openloop.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

bash "$HERE/download_smoke.sh"
FETCH_ONLY=1 python "$HERE/openloop_replay.py"
echo "PASS: open-loop assets cached under ${HF_HOME:-/root/.cache/huggingface}"
