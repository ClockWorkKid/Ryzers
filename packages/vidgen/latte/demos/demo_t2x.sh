#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Text-to-video / text-to-image generation (sample/sample_t2x.py + sample/pipeline_latte.py,
# models/latte_t2v.py; T5 text encoder + SD-VAE).
#   TASK = t2v (16 frames) | t2i (single frame)
# Weights: run scripts/download_checkpoints.sh (WHICH=t2v) first -> $LATTE1_DIR (maxin-cn/Latte-1).
# Overridable: NUM_SAMPLING_STEPS, SAMPLE_METHOD (DDIM|DDPM|...), OUT_DIR, PROMPT (single prompt).
set -euo pipefail

TASK="${TASK:-t2v}"
LATTE1_DIR="${LATTE1_DIR:-/models/Latte-1}"
OUT="${OUT_DIR:-/outputs}/${TASK}"
LATTE_REPO="${LATTE_REPO:-/repos/latte}"
[ "$TASK" = "t2v" ] || [ "$TASK" = "t2i" ] || { echo "unknown TASK=$TASK (t2v|t2i)"; exit 1; }

CONF="$LATTE_REPO/configs/t2x/${TASK}_sample.yaml"
TMP="/tmp/${TASK}_sample.yaml"
cp "$CONF" "$TMP"
sed -i "s#^pretrained_model_path:.*#pretrained_model_path: \"$LATTE1_DIR\"#" "$TMP"
sed -i "s#^save_img_path:.*#save_img_path: \"$OUT/${TASK}-\"#" "$TMP"
[ -n "${NUM_SAMPLING_STEPS:-}" ] && sed -i "s#^num_sampling_steps:.*#num_sampling_steps: $NUM_SAMPLING_STEPS#" "$TMP"
[ -n "${SAMPLE_METHOD:-}" ]      && sed -i "s#^sample_method:.*#sample_method: '$SAMPLE_METHOD'#" "$TMP"
# Optional: replace the built-in prompt list with a single PROMPT (keeps YAML list form).
if [ -n "${PROMPT:-}" ]; then
  python - "$TMP" "$PROMPT" <<'PY'
import sys
from omegaconf import OmegaConf
p, prompt = sys.argv[1], sys.argv[2]
c = OmegaConf.load(p); c.text_prompt = [prompt]; OmegaConf.save(c, p)
PY
fi

mkdir -p "$OUT"
echo "[demo_t2x] TASK=$TASK model=$LATTE1_DIR -> $OUT"
grep -E "num_sampling_steps|sample_method|video_length|image_size|enable_vae_temporal_decoder" "$TMP" || true
cd "$LATTE_REPO"
python sample/sample_t2x.py --config "$TMP"
echo "[demo_t2x] done -> $OUT/"
