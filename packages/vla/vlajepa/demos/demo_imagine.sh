#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Capability 2b: visualize the V-JEPA latent world-model "imagination" on a real
# episode -- predicted vs actual future latents (cosine/L1 curves + PCA token maps).
# NOTE: JEPA predicts latents, not pixels; there is no upstream pixel decoder.
#
#   ryzers run /ryzers/demo_imagine.sh
#   SUITE=libero_object TASK_ID=0 EPISODE=0 T0=0 ryzers run /ryzers/demo_imagine.sh
set -euo pipefail
exec python /ryzers/imagine_latent.py
