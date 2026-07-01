#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# Regenerate the post-ViT (stage-B) random-token-drop latency breakdown PNG from the
# curated latency.csv, using plot_preenc's stacked vision/prefill/flow renderer.
# Run via: ryzer_shell.sh 'bash /scripts/plot_visdrop_latency.sh'
set -euo pipefail
export RAW=/ryzers/outputs/visdrop_raw
export OUT=/ryzers/outputs/visdrop_out
export ACC_TITLE="MolmoAct2-Think LIBERO: accuracy vs random vision-token retention (post-ViT)"
export ACC_XLABEL="vision tokens kept (post-encoder) (%)"
export LAT_TITLE="Post-ViT (stage-B) random token drop: latency decomposition (vision constant, LLM prefill scales)"
export LAT_XLABEL="vision tokens kept (post-encoder)"
mkdir -p "$OUT"
python /scripts/plot_preenc.py
echo "wrote $OUT/latency.png"
