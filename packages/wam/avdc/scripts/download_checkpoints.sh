#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Fetch AVDC weights into the mounted models dir at runtime. Nothing is re-hosted (rule 8): these
# are the upstream-published HF direct links (Po-Chen/flowdiffusion, AVDC_experiments download.sh).
#   WHICH=metaworld     -> Meta-World video model            -> $DEST/metaworld/model-24.pt
#   WHICH=metaworld-DA  -> Meta-World (random-shift aug)     -> $DEST/metaworld_DA/model-24.pt
#   WHICH=ithor         -> iTHOR video model                 -> $DEST/ithor/model-16.pt
#   WHICH=all           -> metaworld + metaworld-DA (default; what the MW demos need)
# CLIP ViT-B/32 is fetched automatically by transformers into the mounted HF cache.
set -euo pipefail

WHICH="${WHICH:-all}"
DEST="${MODELS_DIR:-/models}"
BASE="https://huggingface.co/Po-Chen/flowdiffusion/resolve/main/ckpts"

fetch() {  # url dest
  local url="$1" out="$2"
  if [ -s "$out" ]; then echo "exists, skip -> $out"; return; fi
  mkdir -p "$(dirname "$out")"
  echo "Downloading $url -> $out"
  if command -v wget >/dev/null 2>&1; then wget -c -q --show-progress -O "$out" "$url";
  else curl -fL --progress-bar -o "$out" "$url"; fi
}

case "$WHICH" in
  metaworld)     fetch "$BASE/metaworld/model-24.pt"    "$DEST/metaworld/model-24.pt" ;;
  metaworld-DA)  fetch "$BASE/metaworld_DA/model-24.pt" "$DEST/metaworld_DA/model-24.pt" ;;
  ithor)         fetch "$BASE/ithor/model-16.pt"        "$DEST/ithor/model-16.pt" ;;
  all)           fetch "$BASE/metaworld/model-24.pt"    "$DEST/metaworld/model-24.pt"
                 fetch "$BASE/metaworld_DA/model-24.pt" "$DEST/metaworld_DA/model-24.pt" ;;
  *) echo "unknown WHICH=$WHICH (choose: metaworld | metaworld-DA | ithor | all)"; exit 1 ;;
esac
echo "done -> $DEST"
