#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Fetch Micro-World weights into the mounted models dir (rule 8: nothing is re-hosted). Mirrors the
# upstream README layout so the unmodified examples resolve their paths:
#   $MODELS_DIR/
#     Diffusion_Transformer/Wan2.1-T2V-1.3B     <- Wan-AI/Wan2.1-T2V-1.3B        (t2v + t2w base)
#     Diffusion_Transformer/Wan2.1-I2V-14B-480P <- Wan-AI/Wan2.1-I2V-14B-480P    (i2v + i2w base)
#     T2W                                        <- amd/Micro-World-T2W  (transformer/ + lora)
#     I2W                                        <- amd/Micro-World-I2W  (transformer/ + lora)
#
# WHICH selects what to fetch (base weights are large; already-present dirs are verified + skipped):
#   t2v     -> Wan2.1-T2V-1.3B
#   t2w     -> Wan2.1-T2V-1.3B + amd/Micro-World-T2W        (default)
#   i2v     -> Wan2.1-I2V-14B-480P
#   i2w     -> Wan2.1-I2V-14B-480P + amd/Micro-World-I2W
#   mw-t2w  -> amd/Micro-World-T2W only (base reused separately)
#   mw-i2w  -> amd/Micro-World-I2W only (base reused separately)
#   all     -> everything
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$HERE/_hf_common.sh"

WHICH="${WHICH:-t2w}"
DEST="${MODELS_DIR:-/models}"
DT="$DEST/Diffusion_Transformer"
mkdir -p "$DT"

get_wan_t2v() { hf_prefetch "Wan-AI/Wan2.1-T2V-1.3B"     --local-dir "$DT/Wan2.1-T2V-1.3B"; }
get_wan_i2v() { hf_prefetch "Wan-AI/Wan2.1-I2V-14B-480P" --local-dir "$DT/Wan2.1-I2V-14B-480P"; }
get_mw_t2w()  { hf_prefetch "amd/Micro-World-T2W"        --local-dir "$DEST/T2W"; }
get_mw_i2w()  { hf_prefetch "amd/Micro-World-I2W"        --local-dir "$DEST/I2W"; }

case "$WHICH" in
  t2v)         get_wan_t2v ;;
  t2w)         get_wan_t2v; get_mw_t2w ;;
  i2v)         get_wan_i2v ;;
  i2w)         get_wan_i2v; get_mw_i2w ;;
  mw-t2w)      get_mw_t2w ;;
  mw-i2w)      get_mw_i2w ;;
  all)         get_wan_t2v; get_wan_i2v; get_mw_t2w; get_mw_i2w ;;
  *) echo "unknown WHICH=$WHICH (choose: t2v | t2w | i2v | i2w | mw-t2w | mw-i2w | all)"; exit 1 ;;
esac
echo "done -> $DEST (WHICH=$WHICH)"
