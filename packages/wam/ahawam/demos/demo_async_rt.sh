#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Asynchronous real-time serving demo for AHA-WAM on Strix Halo (gfx1151).
# Reuses the upstream deploy/ stack unchanged: the single-process async policy
# server (deploy/server/wam_policy_server.py --async-mode, AsyncAHAWAMRuntime)
# decouples the slow video-context prefill from the fast action-chunk executor,
# and the async dummy client (deploy/client/wam_policy_async_dummy_client.py)
# drives it over two TCP channels (image push + action_request) and reports the
# end-to-end action latency / throughput.
#
#   ryzers run /ryzers/demos/demo_async_rt.sh                     # AHA-WAM-Flash (1-step)
#   WHICH=robotwin NUM_STEPS=10 ryzers run /ryzers/demos/demo_async_rt.sh   # base AHA-WAM
set -uo pipefail

AHAWAM_REPO=/repos/ahawam
REL=/models/ahawam_release
WHICH="${WHICH:-flash}"                 # flash | robotwin
case "$WHICH" in
  flash)    CKPT="${CKPT:-$REL/robotwin_ahawam-flash.pt}"; DEF_STEPS=1 ;;
  robotwin) CKPT="${CKPT:-$REL/robotwin_ahawam.pt}";       DEF_STEPS=10 ;;
  *) echo "WHICH must be flash|robotwin" >&2; exit 2 ;;
esac
NUM_STEPS="${NUM_STEPS:-$DEF_STEPS}"
STATS="${DATASET_STATS:-$REL/dataset_stats.json}"
INSTRUCTION="${INSTRUCTION:-click the bell}"
PORT="${PORT:-10000}"
NUM_ACTION_REQUESTS="${NUM_ACTION_REQUESTS:-30}"
ACTION_RATE="${ACTION_RATE:-10}"
IMAGE_FPS="${IMAGE_FPS:-30}"
PREFILL_WAIT="${PREFILL_WAIT:-25}"
OUT_DIR="${OUT_DIR:-/outputs}"
SRV_LOG="$OUT_DIR/async_server_${WHICH}.log"
mkdir -p "$OUT_DIR"

export PYTHONPATH="$AHAWAM_REPO/src:$AHAWAM_REPO:${PYTHONPATH:-}"

# Fetch the requested checkpoint if missing (idempotent).
bash /ryzers/scripts/download_checkpoints.sh "$WHICH"
[ -f "$CKPT" ] || { echo "missing $CKPT" >&2; exit 1; }

# Runtime deploy config for the public AHAWAM policy wrapper.
# NOTE: upstream ships configs/deploy.yml, but this Hydra build only resolves the
# .yaml extension, so config_name=deploy is not found. sim_robotwin.yaml composes the
# same base create_ahawam model (proprio/action=14, action_horizon=num_frames-1=64) and
# is the proven eval config, so we use it as the deploy Hydra config here.
DEPLOY_YML="$OUT_DIR/deploy_ahawam_${WHICH}.yml"
cat > "$DEPLOY_YML" <<YML
checkpoint_path: $CKPT
dataset_stats_path: $STATS
project_root: $AHAWAM_REPO
hydra_config_name: sim_robotwin
task: null
device: cuda
mixed_precision: bf16
action_horizon: null
num_inference_steps: $NUM_STEPS
sigma_shift: null
seed: 0
text_cfg_scale: 1.0
negative_prompt: ""
rand_device: cpu
tiled: false
warmup_rounds: 1
video_height: 384
video_width: 320
YML
echo "== deploy config ($WHICH, num_inference_steps=$NUM_STEPS) =="; cat "$DEPLOY_YML"

cleanup() { [ -n "${SRV_PID:-}" ] && kill "$SRV_PID" 2>/dev/null; wait "${SRV_PID:-}" 2>/dev/null; }
trap cleanup EXIT INT TERM

echo "== starting async policy server (pid will follow) =="
python -u "$AHAWAM_REPO/deploy/server/wam_policy_server.py" \
  --async-mode \
  --policy-path "$DEPLOY_YML" \
  --policy-module deploy.server.ahawam_policy \
  --policy-class AHAWAMPolicy \
  --instruction "$INSTRUCTION" \
  --action-dim 14 \
  --host 127.0.0.1 --port "$PORT" > "$SRV_LOG" 2>&1 &
SRV_PID=$!

echo "== waiting for server to bind (model load + warmup) =="
for i in $(seq 1 120); do
  if ! kill -0 "$SRV_PID" 2>/dev/null; then echo "SERVER DIED early; log:"; tail -40 "$SRV_LOG"; exit 1; fi
  if grep -q "Binding TCP server" "$SRV_LOG" 2>/dev/null; then echo "server bound after ${i}s"; break; fi
  sleep 1
done
grep -q "Binding TCP server" "$SRV_LOG" || { echo "server never bound; log:"; tail -60 "$SRV_LOG"; exit 1; }

echo "== running async dummy client =="
python -u "$AHAWAM_REPO/deploy/client/wam_policy_async_dummy_client.py" \
  --host 127.0.0.1 --port "$PORT" \
  --instruction "$INSTRUCTION" \
  --action-dim 14 \
  --image-height 384 --image-width 320 \
  --image-push-fps "$IMAGE_FPS" \
  --action-request-rate "$ACTION_RATE" \
  --num-action-requests "$NUM_ACTION_REQUESTS" \
  --prefill-wait "$PREFILL_WAIT"
CLIENT_RC=$?

echo "== server log (model latency lines) =="
grep -iE "model_latency|action_chunk_shape|prefill|kv_version|Inference output ready" "$SRV_LOG" | tail -20 || true
echo "PASS: async RT demo complete (WHICH=$WHICH steps=$NUM_STEPS rc=$CLIENT_RC)"
exit $CLIENT_RC
