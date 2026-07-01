#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Launch the Route-A group-drop ablation as a DETACHED container so it survives
# SSH/tunnel drops.
#   bash run_groupdrop_detached.sh        # start (resumable)
#   docker logs -f molmoact2_groupdrop    # follow
#   docker stop molmoact2_groupdrop       # stop
# Results stream to: $ROOT/outputs/ablation_groupdrop/{results.csv, run_*/eval_info.json}
set -euo pipefail
ROOT="${STRIX_REMOTE_ROOT:-$HOME/molmoact2-xarm6}"
IMAGE="${RYZER_IMAGE:-molmoact2:latest}"
NAME="${NAME:-molmoact2_groupdrop}"
RENDER_GID="$(getent group render | cut -d: -f3)"
VIDEO_GID="$(getent group video  | cut -d: -f3)"

if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  echo "[groupdrop] container '$NAME' exists; remove it first (docker rm -f $NAME)."
  docker ps -a --filter "name=$NAME" --format '  {{.Names}}  {{.Status}}'
  exit 1
fi

docker run -d --name "$NAME" \
  --shm-size 16G --cap-add=SYS_PTRACE --network=host --ipc=host \
  --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined \
  --group-add "${RENDER_GID:-render}" --group-add "${VIDEO_GID:-video}" \
  -e HSA_OVERRIDE_GFX_VERSION=11.5.1 -e HF_HOME=/root/.cache/huggingface \
  -e HF_HUB_DISABLE_TELEMETRY=1 -e TOKENIZERS_PARALLELISM=false \
  -e N_EP="${N_EP:-1}" -e NUM_STEPS="${NUM_STEPS:-4}" -e SEED="${SEED:-1000}" \
  -e SUITES="${SUITES:-libero_spatial libero_object libero_goal libero_10 libero_90}" \
  -e FRACS="${FRACS:-1.00 0.75 0.50 0.25}" \
  -v "$ROOT/cache/huggingface":/root/.cache/huggingface \
  -v "$ROOT/models":/models \
  -v "$ROOT/datasets":/datasets \
  -v "$ROOT/inputs":/inputs \
  -v "$ROOT/repos":/repos \
  -v "$ROOT/scripts":/scripts \
  -v "$ROOT/outputs":/ryzers/outputs \
  -v "$ROOT/sim":/sim \
  -w /repos "$IMAGE" bash -lc "bash /scripts/sweep_groupdrop.sh"

echo "[groupdrop] launched '$NAME'"
echo "  follow:  docker logs -f $NAME"
echo "  results: $ROOT/outputs/ablation_groupdrop/results.csv"
