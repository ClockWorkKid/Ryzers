#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# Render Route-A group-drop curated plots inside the container (avoids PowerShell
# quoting). Run via: ryzer_shell.sh 'bash /scripts/plot_groupdrop.sh'
set -euo pipefail
export RAW=/ryzers/outputs/ablation_groupdrop
export OUT=/ryzers/outputs/ablation_groupdrop
export ACC_TITLE="MolmoAct2-Think LIBERO: accuracy vs Route-A GROUP-DROP retention (vision+prefill, bf16)"
export ACC_XLABEL="pooling groups kept = vision patches + LLM tokens (%)"
export GRID_TITLE="Route-A group-drop: per-task closed-loop success (1 ep/task; libero_90 = 15-task subsample)"
export LAT_TITLE="Route-A group-drop: latency decomposition (vision AND prefill both scale)"
python /scripts/plot_preenc.py
