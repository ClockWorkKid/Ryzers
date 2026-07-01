#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Real-time (_RT) interactive demo: the LIBERO sim runs at wall-clock speed while the
# MolmoAct2 policy plans asynchronously, so you can see planner latency: the robot
# holds its pose while "thinking", then moves when the next action chunk lands. This is
# the real-time sibling of demo_interactive.sh (which fast-replays a clean chunk rollout).
# Local inference only; host networking (config.yaml) makes it reachable on localhost.
#
#   ryzers run /ryzers/demo_interactive_rt.sh           # Panda, then open http://localhost:8081
#   EMBODIMENT=ur5e ryzers run /ryzers/demo_interactive_rt.sh
#   EMBODIMENT=xarm6 ryzers run /ryzers/demo_interactive_rt.sh
#   RT_HZ=20 RT_LOOKAHEAD=0 ryzers run /ryzers/demo_interactive_rt.sh
#   THINK=1 ryzers run /ryzers/demo_interactive_rt.sh   # full depth reasoning (slower)
#
# Only if you run this on a REMOTE/headless box, forward the port first:
#   ssh -L 8081:localhost:8081 <host>
#
# FAST MODE by default: depth reasoning OFF (THINK=0) + num_steps=4. The overnight
# closed-loop sweep showed this keeps 100% success on libero_object while running
# ~2.9x faster than the full depth-reasoning path (which also adds a ~20s first-plan
# stall), so the real-time view actually flows. Set THINK=1 for the full "Think"
# spatial-reasoning path (slower; better on harder spatial tasks).
set -euo pipefail

export SUITE="${SUITE:-libero_object}"
export TASK_ID="${TASK_ID:-3}"
export SEED="${SEED:-1000}"
export THINK="${THINK:-0}"
export NUM_STEPS="${NUM_STEPS:-4}"
export CKPT="${CKPT:-allenai/MolmoAct2-Think-LIBERO}"
export PORT="${PORT:-8081}"
export VIEW_RES="${VIEW_RES:-720}"
export RT_HZ="${RT_HZ:-20}"
export RT_LOOKAHEAD="${RT_LOOKAHEAD:-0}"
# RTC-style "plan ahead while executing" (research). RT_STITCH=hold keeps the original
# stop-and-decide demo; RT_STITCH=blend pipelines planning and stitches chunks for
# continuous motion. RT_REPLAN_AT=-1 auto-picks ~chunk/2; RT_BLEND_STEPS is the ramp.
export RT_STITCH="${RT_STITCH:-hold}"
export RT_REPLAN_AT="${RT_REPLAN_AT:--1}"
export RT_BLEND_STEPS="${RT_BLEND_STEPS:-4}"
export RT_GRIPPER_HYST="${RT_GRIPPER_HYST:-0.4}"
export PYTHONUNBUFFERED=1

EMBODIMENT="${EMBODIMENT:-}"
case "${EMBODIMENT,,}" in
  ur5e)
    export EMBODIMENT_GRIPPER="${EMBODIMENT_GRIPPER:-Robotiq85Gripper}"
    export EMBODIMENT_UR5E_ROBOTIQ_INIT_BLEND="${EMBODIMENT_UR5E_ROBOTIQ_INIT_BLEND:-0.5}"
    export EMBODIMENT_UR5E_ROBOTIQ_EEF_OFFSET="${EMBODIMENT_UR5E_ROBOTIQ_EEF_OFFSET:-0}"
    export EMBODIMENT_UR5E_ROBOTIQ_CAMERA_OFFSET="${EMBODIMENT_UR5E_ROBOTIQ_CAMERA_OFFSET:-0}"
    ;;
  xarm6)
    export EMBODIMENT_GRIPPER="${EMBODIMENT_GRIPPER:-XArmGripper}"
    export EMBODIMENT_EXECUTOR="${EMBODIMENT_EXECUTOR:-absolute}"
    export EMBODIMENT_SERVO_STEPS="${EMBODIMENT_SERVO_STEPS:-2}"
    export EMBODIMENT_XARM6_CAMERA_MATCH="${EMBODIMENT_XARM6_CAMERA_MATCH:-1}"
    ;;
esac

PY=/opt/libero-venv/bin/python
[ -x "$PY" ] || PY=python
SERVER="${SERVER:-/ryzers/interactive_server_rt.py}"
[ -f "$SERVER" ] || SERVER="$(dirname "$0")/interactive_server_rt.py"

# Load the policy in bf16 (matches the validated DROID bench: ~0.6 s/plan vs ~3.2 s for
# the lerobot loader's default fp32 on gfx1151). This is the deployment-realistic regime
# where async chunk blending can actually hide planner latency. Applies an idempotent
# patch to the installed lerobot loader; override with MOLMOACT2_DTYPE=float32 to revert.
export MOLMOACT2_DTYPE="${MOLMOACT2_DTYPE:-bfloat16}"
DTYPE_PATCH="$(dirname "$SERVER")/apply_dtype_patch.py"
[ -f "$DTYPE_PATCH" ] || DTYPE_PATCH="$(dirname "$0")/apply_dtype_patch.py"
[ -f "$DTYPE_PATCH" ] && "$PY" "$DTYPE_PATCH" || echo "[dtype] patch not found; using loader default"

echo "Real-time demo | arm=${EMBODIMENT:-panda} suite=$SUITE task_id=$TASK_ID seed=$SEED think=$THINK num_steps=$NUM_STEPS port=$PORT hz=$RT_HZ stitch=$RT_STITCH replan_at=$RT_REPLAN_AT blend=$RT_BLEND_STEPS"
echo "Open http://localhost:$PORT in your browser (remote box: ssh -L $PORT:localhost:$PORT <host>)"
if [ -n "$EMBODIMENT" ] && [ "${EMBODIMENT,,}" != "panda" ]; then
  EMB_DIR="$(dirname "$SERVER")"
  [ -f "$EMB_DIR/embodiment.py" ] || EMB_DIR="$(dirname "$0")"
  exec "$PY" -c "import sys, runpy; sys.path.insert(0, '$EMB_DIR'); import embodiment; r=embodiment.apply_from_env(); print('[embodiment] LIBERO arm ->', r, flush=True); runpy.run_path('$SERVER', run_name='__main__')"
fi
exec "$PY" "$SERVER"
