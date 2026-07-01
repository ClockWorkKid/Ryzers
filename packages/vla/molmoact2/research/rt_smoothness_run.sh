#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Driver (runs on the remote host) for the RT chunk-stitching smoothness ablation.
# Launches a one-off molmoact2 container that runs rt_smoothness_ablation.py (sync /
# async-hold / async-blend) and then plot_rt_smoothness.py. Python files are staged in
# $WORK (scp'd from the laptop first); results land under the persistent outputs volume.
#
# Override knobs from the laptop, e.g.:
#   ssh <remote-host> "MODES=blend TASKS=3 RT_MAX_STEPS=120 bash /tmp/rt_smoothness_run.sh"   # smoke
#   ssh <remote-host> "bash /tmp/rt_smoothness_run.sh"                                          # full
set -euo pipefail

IMAGE="${IMAGE:-molmoact2:latest}"
ROOT="${ROOT:-$HOME/Ryzers-fork/workspace}"
WORK="${WORK:-/tmp/rtsmooth}"
OUTSUB="${OUTSUB:-rt_smoothness}"

# Don't fight an in-flight demo for the single GPU.
for c in molmoact2_interactive_rt molmoact2_interactive molmoact2_interactive_xarm6 \
         molmoact2_interactive_ur5e molmoact2_rt_smoothness; do
  docker rm -f "$c" >/dev/null 2>&1 || true
done

RENDER_GID="$(getent group render | cut -d: -f3 || true)"
VIDEO_GID="$(getent group video | cut -d: -f3 || true)"

docker run --rm -i --name molmoact2_rt_smoothness \
  --shm-size 16G --cap-add=SYS_PTRACE --network=host --ipc=host \
  --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined \
  --group-add "${RENDER_GID:-render}" --group-add "${VIDEO_GID:-video}" \
  -e HSA_OVERRIDE_GFX_VERSION=11.5.1 -e HF_HOME=/root/.cache/huggingface \
  -e HF_HUB_DISABLE_TELEMETRY=1 -e TOKENIZERS_PARALLELISM=false \
  -e MUJOCO_GL=egl -e PYOPENGL_PLATFORM=egl \
  -e TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 -e TORCH_BLAS_PREFER_HIPBLASLT=0 \
  -e PYTHONUNBUFFERED=1 \
  -e SUITE="${SUITE:-libero_object}" -e TASKS="${TASKS:-3,2}" -e SEED="${SEED:-1000}" \
  -e THINK="${THINK:-0}" -e NUM_STEPS="${NUM_STEPS:-4}" \
  -e RT_HZ="${RT_HZ:-20}" -e RT_MAX_STEPS="${RT_MAX_STEPS:-400}" \
  -e RT_REPLAN_AT="${RT_REPLAN_AT:--1}" -e RT_BLEND_STEPS="${RT_BLEND_STEPS:-4}" \
  -e RT_GRIPPER_HYST="${RT_GRIPPER_HYST:-0.4}" -e MODES="${MODES:-sync,hold,blend}" \
  -e RT_SERVER=/work/interactive_server_rt.py \
  -e MOLMOACT2_DTYPE="${MOLMOACT2_DTYPE:-bfloat16}" \
  -e OUT_DIR="/outputs/$OUTSUB" \
  -v "$ROOT/.cache/huggingface":/root/.cache/huggingface \
  -v "$ROOT/molmoact2/outputs":/outputs \
  -v "$WORK":/work:ro \
  "$IMAGE" bash -lc "/opt/libero-venv/bin/python /work/apply_dtype_patch.py && \
                     /opt/libero-venv/bin/python /work/rt_smoothness_ablation.py && \
                     /opt/libero-venv/bin/python /work/plot_rt_smoothness.py \
                       --in /outputs/$OUTSUB --out /outputs/$OUTSUB/plots"

echo "=== results on host ==="
ls -1 "$ROOT/molmoact2/outputs/$OUTSUB" 2>/dev/null || true
ls -1 "$ROOT/molmoact2/outputs/$OUTSUB/plots" 2>/dev/null || true
