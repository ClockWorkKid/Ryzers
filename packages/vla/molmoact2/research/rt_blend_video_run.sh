#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Driver (runs on remote host): record the side-by-side async-HOLD vs async-BLEND video.
# Loads the model in bf16 (apply_dtype_patch) so latency matches deployment (~0.6 s/plan),
# which is the regime where blend actually hides planner latency. Files staged in $WORK.
#   ssh <remote-host> "TASK_ID=3 bash /tmp/rt_blend_video_run.sh"
set -euo pipefail

IMAGE="${IMAGE:-molmoact2:latest}"
ROOT="${ROOT:-$HOME/Ryzers-fork/workspace}"
WORK="${WORK:-/tmp/rtsmooth}"
OUTSUB="${OUTSUB:-rt_smoothness}"

for c in molmoact2_interactive_rt molmoact2_interactive molmoact2_rt_smoothness molmoact2_rt_video; do
  docker rm -f "$c" >/dev/null 2>&1 || true
done

RENDER_GID="$(getent group render | cut -d: -f3 || true)"
VIDEO_GID="$(getent group video | cut -d: -f3 || true)"

docker run --rm -i --name molmoact2_rt_video \
  --shm-size 16G --cap-add=SYS_PTRACE --network=host --ipc=host \
  --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined \
  --group-add "${RENDER_GID:-render}" --group-add "${VIDEO_GID:-video}" \
  -e HSA_OVERRIDE_GFX_VERSION=11.5.1 -e HF_HOME=/root/.cache/huggingface \
  -e HF_HUB_DISABLE_TELEMETRY=1 -e TOKENIZERS_PARALLELISM=false \
  -e MUJOCO_GL=egl -e PYOPENGL_PLATFORM=egl \
  -e TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 -e TORCH_BLAS_PREFER_HIPBLASLT=0 \
  -e PYTHONUNBUFFERED=1 \
  -e SUITE="${SUITE:-libero_object}" -e TASK_ID="${TASK_ID:-3}" -e SEED="${SEED:-1000}" \
  -e THINK="${THINK:-0}" -e NUM_STEPS="${NUM_STEPS:-4}" \
  -e RT_HZ="${RT_HZ:-20}" -e RT_MAX_STEPS="${RT_MAX_STEPS:-200}" -e WALL_CAP="${WALL_CAP:-90}" \
  -e RT_REPLAN_AT="${RT_REPLAN_AT:--1}" -e RT_BLEND_STEPS="${RT_BLEND_STEPS:-4}" \
  -e PANEL_RES="${PANEL_RES:-440}" \
  -e RT_SERVER=/work/interactive_server_rt.py -e ABL=/work/rt_smoothness_ablation.py \
  -e MOLMOACT2_DTYPE="${MOLMOACT2_DTYPE:-bfloat16}" \
  -e OUT_DIR="/outputs/$OUTSUB" \
  -v "$ROOT/.cache/huggingface":/root/.cache/huggingface \
  -v "$ROOT/molmoact2/outputs":/outputs \
  -v "$WORK":/work:ro \
  "$IMAGE" bash -lc "/opt/libero-venv/bin/python /work/apply_dtype_patch.py && \
                     /opt/libero-venv/bin/python /work/rt_blend_video.py"

echo "=== videos on host ==="
ls -1t "$ROOT/molmoact2/outputs/$OUTSUB"/blend_vs_hold_*.mp4 2>/dev/null | head
