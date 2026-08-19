#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# FlowWAM *GENUINE* closed-loop RoboTwin 2.0 evaluation on Strix Halo (gfx1151), de-vendored
# simulation/robotwin base. This is the upstream FlowWAM flow-ACTION pipeline (world-action model):
#
#   RoboTwin env --obs(head/left/right RGB + qpos + instruction)--> flow_action_server
#     server: dual-stream DiT generates flow-conditioned video latents -> IDM action expert -> 14-D
#             action chunk  --ws://:8000-->  robotwin_policy client  --> TASK_ENV.take_action / step
#   ...replans every EXECUTE_WINDOW actions until success/timeout. RoboTwin scores success + writes
#   per-episode videos. NO RoboTwin source edits (policy is symlinked into policy/flowwam).
#
# We reuse the upstream server + robotwin_policy VERBATIM; the only project specifics are
# ROCm env, runtime weight/repo fetch, and co-locating server+client in ONE venv (upstream
# splits them across two conda envs to keep torch out of the sim env; we already have both here).
#
# Requires the build chain:  ryzers build robotwin flowwam --name flowwam-robotwin
#   ryzers run /ryzers/demos/demo_closedloop_robotwin.sh
#   TASK=place_dual_shoes NUM_EPISODES=1 ryzers run /ryzers/demos/demo_closedloop_robotwin.sh
set -uo pipefail

if [ ! -d /opt/RoboTwin ]; then
  echo "ERROR: simulation/robotwin base not found (no /opt/RoboTwin)." >&2
  echo "       Build the chain:  ryzers build robotwin flowwam --name flowwam-robotwin" >&2
  exit 1
fi

# ---- config ----
FULL_REPO="${FLOWWAM_FULL_REPO:-/repos/flowwam-full}"     # real FlowWAM (action policy) repo
FLOWWAM_FULL_COMMIT="${FLOWWAM_FULL_COMMIT:-68abaa2}"
MDIR="${FLOWWAM_MODEL_DIR:-/models/flowwam}"
TASK="${TASK:-beat_block_hammer}"
TASK_CONFIG="${TASK_CONFIG:-demo_clean}"
SEED="${SEED:-0}"
GPU_ID="${GPU_ID:-0}"
PORT="${PORT:-8000}"
EXECUTE_WINDOW="${EXECUTE_WINDOW:-25}"       # actions executed per replan (chunk = 1 anchor + this)
NUM_EPISODES="${NUM_EPISODES:-3}"            # eval episodes (upstream full-eval default is 100)
VIDEO_INFERENCE_STEPS="${VIDEO_INFERENCE_STEPS:-25}"
ACTION_INFERENCE_STEPS="${ACTION_INFERENCE_STEPS:-50}"
OUT_DIR="${OUT_DIR:-/outputs/closedloop_robotwin/$TASK}"
CKPT="$MDIR/robotwin/flowwam_robotwin.safetensors"
NORM="$MDIR/robotwin/flowwam_robotwin_action_norm_stats.npz"

# ROCm / SAPIEN-Vulkan env (matches the rest of the flowwam package).
export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.5.1}"
export TORCH_BLAS_PREFER_HIPBLASLT="${TORCH_BLAS_PREFER_HIPBLASLT:-0}"
export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

# ---- 1. fetch the real FlowWAM repo (code clone; idempotent) ----
if [ ! -d "$FULL_REPO/inference" ]; then
  echo "[setup] cloning upstream FlowWAM (action policy) -> $FULL_REPO"
  git clone https://github.com/YixiangChen515/FlowWAM.git "$FULL_REPO"
  git -C "$FULL_REPO" checkout "$FLOWWAM_FULL_COMMIT" 2>/dev/null || true
fi
# The server reads base Wan weights from LOCAL_MODEL_PATH/Wan-AI -> point it at our model cache.
export LOCAL_MODEL_PATH="$MDIR"

# ---- 2. fetch weights (idempotent; never re-hosted) ----
bash /ryzers/scripts/download_checkpoints.sh base
bash /ryzers/scripts/download_checkpoints.sh robotwin
[ -f "$CKPT" ] || { echo "ERROR: missing $CKPT" >&2; exit 1; }

# ---- 2.5 fetch + wire RoboTwin sim assets (embodiments/objects/task_config) into /opt/RoboTwin ----
# One-time (idempotent via the .ready marker). ryzers run already mounts the assets volume and sets
# ROBOTWIN_ASSETS_DIR (inherited from the simulation/robotwin base); this links assets/task_config in
# and routes the aloha-agilex planner curobo->mplib for ROCm. Required: RoboTwin's envs import these.
SETUP="$(ls /ryzers/scripts/setup_robotwin.sh 2>/dev/null || find / -name setup_robotwin.sh 2>/dev/null | head -1)"
[ -n "$SETUP" ] && bash "$SETUP" || echo "WARN: setup_robotwin.sh not found; assets may be missing" >&2

# ---- 3. start the flow-action inference server (background) ----
mkdir -p "$OUT_DIR"
SRV_LOG="$OUT_DIR/server.log"
echo "[server] launching flow_action_server on ws://0.0.0.0:$PORT (log: $SRV_LOG)"
# Give the server the REAL FlowWAM repo's diffsynth (action expert / dual-stream) WITHOUT replacing
# the installed WorldArena diffsynth that the open-loop world-model eval imports. Runtime path
# precedence avoids a second venv; if a build-time import conflict surfaces, split into its own venv.
#
# gfx1151 default-route speedups (quality-preserving): route the
# upstream start_server.sh's PYTHON through opt_launch.py, which arms the bit-exact cross-attn K/V +
# text-embedding cache and torch.compile of the dual-stream block fn BEFORE running the upstream
# flow_action_server.py verbatim. Disable with FLOWWAM_OPT=0 (or FLOWWAM_CACHE=0 / FLOWWAM_COMPILE=0).
CHECKPOINT="$CKPT" ACTION_NORM_PATH="$NORM" LOCAL_MODEL_PATH="$MDIR" \
  PYTHONPATH="$FULL_REPO:${PYTHONPATH:-}" \
  PYTHON="${FLOWWAM_PYTHON:-/ryzers/scripts/opt_python.sh}" \
  HOST=0.0.0.0 PORT="$PORT" DEVICE=cuda \
  VIDEO_INFERENCE_STEPS="$VIDEO_INFERENCE_STEPS" ACTION_INFERENCE_STEPS="$ACTION_INFERENCE_STEPS" \
  bash "$FULL_REPO/inference/start_server.sh" > "$SRV_LOG" 2>&1 &
SRV_PID=$!
trap 'kill "$SRV_PID" 2>/dev/null || true' EXIT

# ---- 4. wait until the server is serving (cap 20 min: first run compiles ROCm kernels) ----
echo "[server] waiting for readiness ..."
for i in $(seq 1 240); do
  if grep -qiE "serving on ws|Uvicorn running|server (ready|started)|Listening" "$SRV_LOG" 2>/dev/null; then
    echo "[server] ready after ${i}x5s"; break
  fi
  if ! kill -0 "$SRV_PID" 2>/dev/null; then echo "ERROR: server died; tail:" >&2; tail -n 30 "$SRV_LOG" >&2; exit 1; fi
  sleep 5
done

# ---- 5. run RoboTwin eval via the upstream policy plugin + stock eval_policy.py ----
# Mirrors upstream inference/robotwin_policy/eval.sh (symlink policy/flowwam + stock eval_policy.py;
# NO RoboTwin source edits) but exposes NUM_EPISODES (upstream eval.sh hardcodes the full 100-ep run).
echo "########## FlowWAM closed-loop RoboTwin $TASK ($TASK_CONFIG, seed=$SEED, $NUM_EPISODES ep) ##########"
ln -sfn "$FULL_REPO/inference/robotwin_policy" /opt/RoboTwin/policy/flowwam
export no_proxy="127.0.0.1,localhost,0.0.0.0,${no_proxy:-}"; export NO_PROXY="$no_proxy"
( cd /opt/RoboTwin && PYTHONWARNINGS=ignore::UserWarning python script/eval_policy.py \
    --config policy/flowwam/deploy_policy.yml --overrides \
    --task_name "$TASK" --task_config "$TASK_CONFIG" --ckpt_setting flowwam --seed "$SEED" \
    --policy_name flowwam --server_host 0.0.0.0 --server_port "$PORT" \
    --action_chunk_size "$((1 + EXECUTE_WINDOW))" --eval_num_episodes "$NUM_EPISODES" ) \
  2>&1 | grep -viE 'svulkan2|Failed to initialize denoiser|cudaErrorInsufficientDriver'
RC=$?

# ---- 6. collect RoboTwin's per-episode videos + result files ----
RES="/opt/RoboTwin/eval_result/$TASK/flowwam/$TASK_CONFIG"
if [ -d "$RES" ]; then cp -r "$RES"/* "$OUT_DIR"/ 2>/dev/null || true; fi
echo "[done] rc=$RC  results -> $OUT_DIR"
exit "$RC"
