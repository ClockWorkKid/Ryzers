#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Weight-free smoke: ROCm torch on the iGPU + flash_attn SDPA shim + the pristine
# groot.vla.model.dreamzero import graph + a tiny AttentionModule forward.
#   ryzers run /ryzers/demos/demo_smoke.sh
set -euo pipefail
exec python /ryzers/test.py
