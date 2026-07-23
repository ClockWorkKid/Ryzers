#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Env sign-of-life for the FlowWAM model layer (no weights). Proves ROCm torch + the DiffSynth
# Wan dual-stream pipeline + SAPIEN/flow deps import and the SDPA attention fallback runs.
#   ryzers run --name flowwam-robotwin /ryzers/demos/demo_smoke.sh
set -euo pipefail
exec python /ryzers/test.py
