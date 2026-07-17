#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Fetch a released NanoWM checkpoint (HF org `knightnemo`) into the mounted HF cache.
# Nothing is re-hosted in the repo (rule 8). Default = smallest DINO-WM domain (PushT).
#
# DRAFT: confirm exact HF layout (checkpoint + matching latent codec) during Phase 2.
set -euo pipefail

DOMAIN="${DOMAIN:-dino_wm_pusht}"
DEST="${RESULTS_DIR:-/models/results}"
mkdir -p "$DEST"

declare -A REPOS=(
  [dino_wm_point_maze]="knightnemo/nanowm-b2-dino-wm-point-maze-30k"
  [dino_wm_wall]="knightnemo/nanowm-b2-dino-wm-wall-15k"
  [dino_wm_rope]="knightnemo/nanowm-b2-dino-wm-rope-15k"
  [dino_wm_granular]="knightnemo/nanowm-b2-dino-wm-granular-15k"
  [dino_wm_pusht]="knightnemo/nanowm-b2-dino-wm-pusht-100k"
  [rt1]="knightnemo/nanowm-b2-rt1-300k"
  [csgo]="knightnemo/nanowm-l2-csgo-100k"
)

REPO="${REPOS[$DOMAIN]:-}"
[ -z "$REPO" ] && { echo "unknown DOMAIN=$DOMAIN; choose one of: ${!REPOS[*]}"; exit 1; }

echo "Downloading $REPO -> $DEST/$DOMAIN"
hf download "$REPO" --local-dir "$DEST/$DOMAIN" ${HF_TOKEN:+--token "$HF_TOKEN"}
# rollout.py's find_model() uses torch.load, so materialize a .pt from the shipped safetensors.
python - "$DEST/$DOMAIN" <<'PY'
import sys, os, torch
from safetensors.torch import load_file
d = sys.argv[1]; st = os.path.join(d, "model.safetensors")
if os.path.isfile(st):
    torch.save(load_file(st), os.path.join(d, "model.pt"))
    print("converted model.safetensors -> model.pt")
PY
echo "done. (latent codec weights, e.g. SD-VAE, download on first rollout run)"
