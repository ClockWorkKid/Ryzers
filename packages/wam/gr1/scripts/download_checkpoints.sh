#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Fetch GR-1 weights into the mounted models dir at runtime. Nothing is re-hosted (rule 8):
# these are the upstream-published direct links (README "CALVIN Benchmark Evaluation").
#   WHICH=mae   -> MAE ViT-B/16 pre-train (facebookresearch/mae)      -> $DEST/mae_pretrain_vit_base.pth
#   WHICH=abcd  -> GR-1 ABCD-D policy snapshot (ByteDance)            -> $DEST/snapshot_ABCD.pt
#   WHICH=abc   -> GR-1 ABC-D  policy snapshot (ByteDance)            -> $DEST/snapshot_ABC.pt
#   WHICH=all   -> mae + abcd (default; what the open-loop demo needs)
# CLIP ViT-B/32 is fetched automatically by clip.load(...) into the mounted HF/torch cache.
set -euo pipefail

WHICH="${WHICH:-all}"
DEST="${MODELS_DIR:-/models}"
mkdir -p "$DEST"

MAE_URL="https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth"
ABCD_URL="https://lf-robot-opensource.bytetos.com/obj/lab-robot-public/gr1_code_release/snapshot_ABCD.pt"
ABC_URL="https://lf-robot-opensource.bytetos.com/obj/lab-robot-public/gr1_code_release/snapshot_ABC.pt"

fetch() {  # url dest
  local url="$1" out="$2"
  if [ -s "$out" ]; then echo "exists, skip -> $out"; return; fi
  echo "Downloading $url -> $out"
  if command -v wget >/dev/null 2>&1; then wget -q --show-progress -O "$out" "$url";
  else curl -fL --progress-bar -o "$out" "$url"; fi
}

case "$WHICH" in
  mae)        fetch "$MAE_URL"  "$DEST/mae_pretrain_vit_base.pth" ;;
  abcd)       fetch "$ABCD_URL" "$DEST/snapshot_ABCD.pt" ;;
  abc)        fetch "$ABC_URL"  "$DEST/snapshot_ABC.pt" ;;
  all)        fetch "$MAE_URL"  "$DEST/mae_pretrain_vit_base.pth"
              fetch "$ABCD_URL" "$DEST/snapshot_ABCD.pt" ;;
  *) echo "unknown WHICH=$WHICH (choose: mae | abcd | abc | all)"; exit 1 ;;
esac
echo "done -> $DEST"
