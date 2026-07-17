#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Fetch Latte weights into the mounted HF cache / models dir. Nothing is re-hosted (rule 8).
#   WHICH=t2v   -> maxin-cn/Latte-1  (T2V/T2I: transformer + T5 text_encoder + SD-VAE + scheduler)
#   WHICH=class -> maxin-cn/Latte    (class-conditional ffs/sky/ucf101/taichi DiT checkpoints)
set -euo pipefail

WHICH="${WHICH:-t2v}"
DEST="${MODELS_DIR:-/models}"
mkdir -p "$DEST"

case "$WHICH" in
  t2v|t2i|latte1)
    REPO="maxin-cn/Latte-1";  OUT="$DEST/Latte-1" ;;
  class|latte0)
    REPO="maxin-cn/Latte";    OUT="$DEST/Latte"   ;;
  *) echo "unknown WHICH=$WHICH (choose: t2v | class)"; exit 1 ;;
esac

# NOTE: the image pins huggingface_hub==0.20.3 (diffusers 0.24.0 compat) -> use huggingface-cli.
echo "Downloading $REPO -> $OUT"
if [ "$WHICH" = "class" ] || [ "$WHICH" = "latte0" ]; then
  # class-conditional: only the 4 XL/2 checkpoints + SD-VAE (skip B/L/S + extra vae dirs).
  huggingface-cli download "$REPO" \
    --include "ffs.pt" "skytimelapse.pt" "taichi-hd.pt" "ucf101.pt" "vae/*" \
    --local-dir "$OUT" ${HF_TOKEN:+--token "$HF_TOKEN"}
else
  # T2V/T2I: everything except the standalone .pt (transformer/ subfolder is what's loaded).
  huggingface-cli download "$REPO" --exclude "t2v_v20240523.pt" \
    --local-dir "$OUT" ${HF_TOKEN:+--token "$HF_TOKEN"}
fi
echo "done -> $OUT"
