#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# RT interactive demo (port 8081) with Route-A group-drop (vision+token drop) baked in.
# Same as /tmp/interactive_rt_run.sh but forwards VIS_GROUPDROP_KEEP_FRAC so the planner
# drops whole 2x2 pooling groups pre-encoder. Defaults: keep=0.5 + RT_STITCH=blend.
# The Ryzers-fork HF cache used here must already carry the group-drop hub patch.
#   ssh <remote-host> "bash /tmp/interactive_rt_groupdrop_run.sh"            # keep=0.5, blend
#   ssh <remote-host> "GD_KEEP=0.75 RT_STITCH=hold bash /tmp/interactive_rt_groupdrop_run.sh"
set -uo pipefail
IMAGE=molmoact2:latest
ROOT="$HOME/Ryzers-fork/workspace"
DEMO="$HOME/interactive_demo"
PORT="${PORT:-8081}"
GD_KEEP="${GD_KEEP:-0.5}"
RENDER_GID="$(getent group render | cut -d: -f3)"
VIDEO_GID="$(getent group video | cut -d: -f3)"

docker rm -f molmoact2_interactive_rt >/dev/null 2>&1 || true

docker run --rm -i --name molmoact2_interactive_rt \
  --shm-size 16G --cap-add=SYS_PTRACE --network=host --ipc=host \
  --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined \
  --group-add "${RENDER_GID:-render}" --group-add "${VIDEO_GID:-video}" \
  -e HSA_OVERRIDE_GFX_VERSION=11.5.1 -e HF_HOME=/root/.cache/huggingface \
  -e HF_HUB_DISABLE_TELEMETRY=1 -e TOKENIZERS_PARALLELISM=false \
  -e MUJOCO_GL=egl -e PYOPENGL_PLATFORM=egl \
  -e TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 -e TORCH_BLAS_PREFER_HIPBLASLT=0 \
  -e PYTHONUNBUFFERED=1 -e PORT="$PORT" \
  -e SUITE="${SUITE:-libero_object}" -e TASK_ID="${TASK_ID:-3}" -e SEED="${SEED:-1000}" -e THINK="${THINK:-0}" \
  -e NUM_STEPS="${NUM_STEPS:-4}" \
  -e VIEW_RES="${VIEW_RES:-720}" -e VIDEO_RES="${VIDEO_RES:-600}" -e RT_HZ="${RT_HZ:-20}" \
  -e RT_LOOKAHEAD="${RT_LOOKAHEAD:-0}" -e OUT_DIR=/outputs \
  -e RT_STITCH="${RT_STITCH:-blend}" -e RT_REPLAN_AT="${RT_REPLAN_AT:--1}" \
  -e RT_BLEND_STEPS="${RT_BLEND_STEPS:-4}" -e RT_GRIPPER_HYST="${RT_GRIPPER_HYST:-0.4}" \
  -e MOLMOACT2_DTYPE="${MOLMOACT2_DTYPE:-bfloat16}" \
  -e VIS_GROUPDROP_KEEP_FRAC="$GD_KEEP" \
  -e SERVER=/demo/interactive_server_rt.py \
  -v "$ROOT/.cache/huggingface":/root/.cache/huggingface \
  -v "$ROOT/molmoact2/outputs":/outputs \
  -v "$DEMO":/demo:ro \
  "$IMAGE" bash -lc "bash /demo/demo_interactive_rt.sh"
