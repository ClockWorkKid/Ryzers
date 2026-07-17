#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Class-conditional / unconditional video generation (sample/sample.py + diffusion/ DDPM/DDIM).
#   DATASET = ucf101 (class-cond, 101 classes) | ffs | sky | taichi (unconditional)
# Weights: run scripts/download_checkpoints.sh (WHICH=class) first -> $LATTE_DIR.
# Overridable: NUM_SAMPLING_STEPS, SAMPLE_METHOD (ddpm|ddim), SEED, OUT_DIR.
set -euo pipefail

DATASET="${DATASET:-ucf101}"
LATTE_DIR="${LATTE_DIR:-/models/Latte}"
OUT="${OUT_DIR:-/outputs}/class_${DATASET}"
LATTE_REPO="${LATTE_REPO:-/repos/latte}"

declare -A CKPTS=([ffs]=ffs.pt [sky]=skytimelapse.pt [taichi]=taichi-hd.pt [ucf101]=ucf101.pt)
CKPT_FILE="${CKPTS[$DATASET]:-}"
[ -z "$CKPT_FILE" ] && { echo "unknown DATASET=$DATASET (ffs|sky|taichi|ucf101)"; exit 1; }

CONF="$LATTE_REPO/configs/$DATASET/${DATASET}_sample.yaml"
TMP="/tmp/${DATASET}_sample.yaml"
cp "$CONF" "$TMP"
sed -i "s#^pretrained_model_path:.*#pretrained_model_path: \"$LATTE_DIR\"#" "$TMP"
[ -n "${NUM_SAMPLING_STEPS:-}" ] && sed -i "s#^num_sampling_steps:.*#num_sampling_steps: $NUM_SAMPLING_STEPS#" "$TMP"
[ -n "${SAMPLE_METHOD:-}" ]      && sed -i "s#^sample_method:.*#sample_method: '$SAMPLE_METHOD'#" "$TMP"
[ -n "${SEED:-}" ]               && sed -i "s#^seed:.*#seed: $SEED#" "$TMP"

mkdir -p "$OUT"
echo "[demo_class] DATASET=$DATASET ckpt=$LATTE_DIR/$CKPT_FILE -> $OUT"
grep -E "num_sampling_steps|sample_method|extras|image_size|num_frames" "$TMP" || true
cd "$LATTE_REPO"
python sample/sample.py --config "$TMP" --ckpt "$LATTE_DIR/$CKPT_FILE" --save_video_path "$OUT"
echo "[demo_class] done -> $OUT/sample.mp4"
