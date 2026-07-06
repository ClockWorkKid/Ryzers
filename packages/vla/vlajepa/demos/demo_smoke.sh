#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Capability 1: full-model smoke. Load the real VLA-JEPA checkpoint (+ Qwen3-VL
# base VLM + V-JEPA2 encoder) and run one action prediction on ROCm. First run
# downloads the weights into the mounted HF cache.
#   ryzers run /ryzers/demo_smoke.sh
set -euo pipefail
exec python /ryzers/model_smoke.py
