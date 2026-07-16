#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Environment sign-of-life (no weights): ROCm torch + gfx1151 GPU + full VERA import.
#   ryzers run /ryzers/demos/demo_smoke.sh
set -euo pipefail
exec python /ryzers/test.py
