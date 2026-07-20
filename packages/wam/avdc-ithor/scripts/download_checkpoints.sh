#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Fetch the AVDC iTHOR video-model checkpoint into the mounted models dir at runtime. Nothing is
# re-hosted (rule 8): this is the upstream-published HF direct link (Po-Chen/flowdiffusion). CLIP
# ViT-B/32 is fetched automatically by transformers into the mounted HF cache.
#   -> $DEST/ithor/model-16.pt
set -euo pipefail

DEST="${MODELS_DIR:-/models}"
BASE="https://huggingface.co/Po-Chen/flowdiffusion/resolve/main/ckpts"
OUT="$DEST/ithor/model-16.pt"

if [ -s "$OUT" ]; then echo "exists, skip -> $OUT"; exit 0; fi
mkdir -p "$(dirname "$OUT")"
echo "Downloading $BASE/ithor/model-16.pt -> $OUT"
if command -v wget >/dev/null 2>&1; then wget -c -q --show-progress -O "$OUT" "$BASE/ithor/model-16.pt";
else curl -fL --progress-bar -o "$OUT" "$BASE/ithor/model-16.pt"; fi
echo "done -> $OUT"
