#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Launch the attention-feedback closed-loop smoke as a DETACHED container so it
# survives SSH/tunnel drops.
#   bash run_attnfb_smoke_detached.sh
#   docker logs -f molmoact2_attnfb_smoke
# Results: $ROOT/outputs/attnfb_smoke/run_<suite>_<cond>.log
set -euo pipefail
ROOT="${STRIX_REMOTE_ROOT:-$HOME/molmoact2-xarm6}"
IMAGE="${RYZER_IMAGE:-molmoact2:latest}"
NAME="${NAME:-molmoact2_attnfb_smoke}"
RENDER_GID="$(getent group render | cut -d: -f3)"
VIDEO_GID="$(getent group video  | cut -d: -f3)"

if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  echo "[attnfb] removing existing container '$NAME'"
  docker rm -f "$NAME" >/dev/null
fi

docker run -d --name "$NAME" \
  --shm-size 16G --cap-add=SYS_PTRACE --network=host --ipc=host \
  --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined \
  --group-add "${RENDER_GID:-render}" --group-add "${VIDEO_GID:-video}" \
  -e HSA_OVERRIDE_GFX_VERSION=11.5.1 -e HF_HOME=/root/.cache/huggingface \
  -e HF_HUB_DISABLE_TELEMETRY=1 -e TOKENIZERS_PARALLELISM=false \
  -e N_EP="${N_EP:-1}" -e SEED="${SEED:-1000}" -e KEEP="${KEEP:-0.5}" \
  -e SUITE="${SUITE:-libero_object}" -e RESET="${RESET:-1}" \
  -e CONDS="${CONDS:-full random attnfb}" -e SURVEY_EVERY="${SURVEY_EVERY:-0}" \
  -v "$ROOT/cache/huggingface":/root/.cache/huggingface \
  -v "$ROOT/models":/models \
  -v "$ROOT/datasets":/datasets \
  -v "$ROOT/inputs":/inputs \
  -v "$ROOT/repos":/repos \
  -v "$ROOT/scripts":/scripts \
  -v "$ROOT/outputs":/ryzers/outputs \
  -v "$ROOT/sim":/sim \
  -w /repos "$IMAGE" bash -lc "bash /scripts/smoke_attnfb_libero.sh"

echo "[attnfb] launched '$NAME'"
echo "  follow:  docker logs -f $NAME"
echo "  results: $ROOT/outputs/attnfb_smoke/"
