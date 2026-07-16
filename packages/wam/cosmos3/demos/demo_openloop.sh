#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Open-loop DROID eval: fetch a few Cosmos3-DROID episodes + the policy checkpoint, replay them
# and compare predicted vs GT actions. GATED: export HF_TOKEN + accept the license.
#   HF_TOKEN=... NUM_EPISODES=5 ryzers run /ryzers/demos/demo_openloop.sh
set -euo pipefail
bash /ryzers/scripts/download_checkpoints.sh
bash /ryzers/scripts/download_datasets.sh
exec python /ryzers/scripts/openloop_replay.py
