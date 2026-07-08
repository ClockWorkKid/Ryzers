#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Download option for the LIBERO-Plus perturbation asset pack. The base image bakes these
# assets at build time; this demo (re)fetches them from the upstream HuggingFace dataset
# and unzips them into the LIBERO-Plus assets tree. Idempotent (skips if present; FORCE=1
# to re-download).
#   ryzers run --name sim-libero-plus /ryzers/demos/demo_download.sh
set -euo pipefail
exec bash /ryzers/scripts/download_assets.sh
