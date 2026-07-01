#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Launch the vision-token ablation as a DETACHED container on the Strix Halo
# host so it survives SSH/tunnel drops. Mirrors ryzer_shell.sh mounts/flags.
#   bash run_ablation_detached.sh           # start (resumable)
#   docker logs -f molmoact2_ablation       # follow
#   docker stop molmoact2_ablation          # stop
# Results stream to: $ROOT/outputs/ablation_visdrop/{results.csv,latency.csv}
set -euo pipefail
ROOT="${STRIX_REMOTE_ROOT:-$HOME/molmoact2-xarm6}"
IMAGE="${RYZER_IMAGE:-molmoact2:latest}"
NAME="${NAME:-molmoact2_ablation}"
RENDER_GID="$(getent group render | cut -d: -f3)"
VIDEO_GID="$(getent group video  | cut -d: -f3)"

if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  echo "[ablation] container '$NAME' exists; remove it first (docker rm -f $NAME) or it is already running."
  docker ps -a --filter "name=$NAME" --format '  {{.Names}}  {{.Status}}'
  exit 1
fi

docker run -d --name "$NAME" \
  --shm-size 16G --cap-add=SYS_PTRACE --network=host --ipc=host \
  --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined \
  --group-add "${RENDER_GID:-render}" --group-add "${VIDEO_GID:-video}" \
  -e HSA_OVERRIDE_GFX_VERSION=11.5.1 -e HF_HOME=/root/.cache/huggingface \
  -e HF_HUB_DISABLE_TELEMETRY=1 -e TOKENIZERS_PARALLELISM=false \
  -e N_EP="${N_EP:-4}" -e NUM_STEPS="${NUM_STEPS:-4}" -e SEED="${SEED:-1000}" \
  -e SUITES="${SUITES:-libero_spatial libero_object libero_goal libero_10 libero_90}" \
  -e FRACS="${FRACS:-0.05 0.10 0.25 0.50 0.75 1.00}" \
  -v "$ROOT/cache/huggingface":/root/.cache/huggingface \
  -v "$ROOT/models":/models \
  -v "$ROOT/datasets":/datasets \
  -v "$ROOT/inputs":/inputs \
  -v "$ROOT/repos":/repos \
  -v "$ROOT/scripts":/scripts \
  -v "$ROOT/outputs":/ryzers/outputs \
  -v "$ROOT/sim":/sim \
  -w /repos "$IMAGE" bash -lc "bash /scripts/sweep_ablation.sh"

echo "[ablation] launched '$NAME'"
echo "  follow:  docker logs -f $NAME"
echo "  results: $ROOT/outputs/ablation_visdrop/results.csv"
