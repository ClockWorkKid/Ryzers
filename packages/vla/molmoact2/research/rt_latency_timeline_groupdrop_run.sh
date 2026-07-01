#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# RT latency-overlap timeline (async-HOLD vs async-BLEND) with Route-A group-drop
# (vision + token drop) baked in. Applies patch_vis_groupdrop.py to the RT HF cache
# hub snapshot, clears the transformers_modules cache so it regenerates patched,
# then runs rt_latency_timeline.py with VIS_GROUPDROP_KEEP_FRAC set.
#   ssh <remote-host> "GD_KEEP=0.5 TASK_ID=3 bash /tmp/rtsmooth/rt_latency_timeline_groupdrop_run.sh"
set -euo pipefail

IMAGE="${IMAGE:-molmoact2:latest}"
ROOT="${ROOT:-$HOME/Ryzers-fork/workspace}"
WORK="${WORK:-/tmp/rtsmooth}"
OUTSUB="${OUTSUB:-rt_smoothness_groupdrop}"
GD_KEEP="${GD_KEEP:-0.5}"

for c in molmoact2_interactive_rt molmoact2_rt_smoothness molmoact2_rt_video molmoact2_rt_latency; do
  docker rm -f "$c" >/dev/null 2>&1 || true
done

RENDER_GID="$(getent group render | cut -d: -f3 || true)"
VIDEO_GID="$(getent group video | cut -d: -f3 || true)"

docker run --rm -i --name molmoact2_rt_latency \
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
  -e RT_HZ="${RT_HZ:-20}" -e MAX_STEPS="${MAX_STEPS:-80}" -e WALL_CAP="${WALL_CAP:-60}" \
  -e RT_REPLAN_AT="${RT_REPLAN_AT:--1}" -e RT_BLEND_STEPS="${RT_BLEND_STEPS:-4}" \
  -e RT_GRIPPER_HYST="${RT_GRIPPER_HYST:-0.4}" \
  -e RT_SERVER=/work/interactive_server_rt.py -e ABL=/work/rt_smoothness_ablation.py \
  -e MOLMOACT2_DTYPE="${MOLMOACT2_DTYPE:-bfloat16}" \
  -e VIS_GROUPDROP_KEEP_FRAC="$GD_KEEP" -e VIS_GROUPDROP_DEBUG=1 \
  -e OUT_DIR="/outputs/$OUTSUB" \
  -v "$ROOT/.cache/huggingface":/root/.cache/huggingface \
  -v "$ROOT/molmoact2/outputs":/outputs \
  -v "$WORK":/work:ro \
  "$IMAGE" bash -lc "/opt/libero-venv/bin/python /work/patch_vis_groupdrop.py /root/.cache/huggingface/hub && \
                     rm -rf /root/.cache/huggingface/modules/transformers_modules/*593d25* \
                            /root/.cache/huggingface/modules/transformers_modules/*d8c1abd8* && \
                     /opt/libero-venv/bin/python /work/apply_dtype_patch.py && \
                     /opt/libero-venv/bin/python /work/rt_latency_timeline.py"

echo "=== latency artifacts on host ==="
ls -1t "$ROOT/molmoact2/outputs/$OUTSUB"/latency_timeline_*.png 2>/dev/null | head
