#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# VERA MimicGen closed-loop via the split model/sim seam (this image is built FROM
# simulation/mimicgen). VERA serves the WAN-1.3B planner + Jacobian IDM over the websocket
# protocol; the sim base's sim_mimicgen harness drives the robosuite/MuJoCo env against it
# through POLICY_FACTORY=vera_mimicgen_policy:build_policy. This is the successor to
# demo_mimicgen.sh (which called the upstream in-package controller directly); it exercises the
# packaged model/simulator boundary end to end.
#   1) start_vera_server --embodiment mimicgen --algo-config <cotracker> --sample-steps N  (bg; vis @ VIS_PORT)
#   2) wait for the policy port
#   3) sim_mimicgen harness rollout with the VERA remote policy  (fg; EGL, osmesa retry)
#   4) save_vis_video -> $OUT_DIR/mimicgen_vis.mp4
# Weights: download_checkpoints.sh wave1; datasets: sim base download_mimicgen_datasets.sh (rule 8).
set -uo pipefail

: "${PORT:=8800}"
: "${VIS_PORT:=8801}"
: "${TASK:=stack_d0}"
: "${MIMICGEN_DATASET_ROOT:=/sim_data/mimicgen_datasets}"
: "${DATASET:=$MIMICGEN_DATASET_ROOT/core/${TASK}.hdf5}"
# Nest under a task subdir explicitly: the chained run script exports a fixed OUT_DIR=/outputs
# (both the sim base and vera set it, last wins), so a bare `: "${OUT_DIR:=...}"` default would
# never apply and every task would dump into /outputs root. Build the path from OUT_DIR instead.
OUT_DIR="${OUT_DIR:-/outputs}/vera_mimicgen/${TASK}"
: "${NUM_DEMOS:=3}"
: "${ROLLOUT_HORIZON:=400}"
: "${RENDER_SIZE:=128}"
: "${SAMPLE_STEPS:=40}"
: "${VERA_WAN_CKPT_ROOT:=/models/wan2.1-t2v-1.3b}"
: "${VERA_MIMICGEN_CKPT_DIR:=/models/vera-ckpts/mimicgen-wan-1.3b}"
: "${VERA_MIMICGEN_DYNAMICS_CKPT:=/models/vera-ckpts/idm-mimicgen-285ouq1q/model.ckpt}"
ALGO_CONFIG="${ALGO_CONFIG:-$VERA_MIMICGEN_CKPT_DIR/algo_config.yaml}"
export VERA_WAN_CKPT_ROOT VERA_MIMICGEN_CKPT_DIR VERA_MIMICGEN_DYNAMICS_CKPT
mkdir -p "$OUT_DIR"

if [ ! -f "$DATASET" ]; then
  echo "[demo_closedloop_mimicgen] dataset not found: $DATASET"
  echo "[demo_closedloop_mimicgen] fetch it first (sim base): ryzers run /ryzers/scripts/download_mimicgen_datasets.sh $TASK"
  exit 1
fi

# cotracker tracker backend on a COPY of the shipped algo_config (never edit the downloaded asset);
# the patched tracker_backend_from_cfg makes MotionTrackConfig.backend take effect (see docs).
PATCHED_ALGO="$OUT_DIR/algo_config.cotracker.yaml"
cp "$ALGO_CONFIG" "$PATCHED_ALGO"
if ! grep -qE '^tracker:' "$PATCHED_ALGO"; then
  {
    echo ""
    echo "tracker:"
    echo "  backend: cotracker"
    echo "  enabled: true"
    echo "  return_visualization: true"
  } >> "$PATCHED_ALGO"
fi
ALGO_CONFIG="$PATCHED_ALGO"

echo "[demo_closedloop_mimicgen] TASK=$TASK DATASET=$DATASET"
echo "[demo_closedloop_mimicgen] starting VERA policy server (port=$PORT vis=$VIS_PORT steps=$SAMPLE_STEPS)"
python -m vera.server.start_vera_server --embodiment mimicgen --port "$PORT" --vis-port "$VIS_PORT" \
  --algo-config "$ALGO_CONFIG" --sample-steps "$SAMPLE_STEPS" \
  > "$OUT_DIR/mimicgen_server.log" 2>&1 &
SRV=$!
cleanup() { echo "[demo_closedloop_mimicgen] stopping server ($SRV)"; kill "$SRV" 2>/dev/null || true; }
trap cleanup EXIT

echo "[demo_closedloop_mimicgen] waiting for policy port $PORT (WAN+IDM load is slow, ~minutes) ..."
for i in $(seq 1 200); do
  if ! kill -0 "$SRV" 2>/dev/null; then echo "[demo_closedloop_mimicgen] server died early:"; tail -n 60 "$OUT_DIR/mimicgen_server.log"; exit 1; fi
  if python -c "import socket,sys; s=socket.socket(); s.settimeout(2); sys.exit(0 if s.connect_ex(('127.0.0.1',$PORT))==0 else 1)"; then
    echo "[demo_closedloop_mimicgen] policy port up"; break
  fi
  sleep 5
done

# Drive the sim base harness (env + rollout) with the VERA remote policy pointed at the server.
run_client() {  # $1 = GL backend
  echo "[demo_closedloop_mimicgen] rollout with MUJOCO_GL=$1 (POLICY_FACTORY=vera_mimicgen_policy:build_policy)"
  MUJOCO_GL="$1" PYOPENGL_PLATFORM="$1" \
  POLICY_FACTORY="vera_mimicgen_policy:build_policy" POLICY_HOST=127.0.0.1 POLICY_PORT="$PORT" \
  RENDER_SIZE="$RENDER_SIZE" \
    python -u -m sim_mimicgen.sanity \
      --dataset "$DATASET" --output-dir "$OUT_DIR" \
      --num-demos "$NUM_DEMOS" --rollout-horizon "$ROLLOUT_HORIZON" --render-size "$RENDER_SIZE" \
      --run-tag vera_remote_eval
}
run_client egl; RC=$?
if [ "$RC" -ne 0 ]; then
  echo "[demo_closedloop_mimicgen] EGL path failed (rc=$RC); retrying with software osmesa"
  run_client osmesa; RC=$?
fi

echo "[demo_closedloop_mimicgen] dumping viewer buffer -> $OUT_DIR/mimicgen_vis.mp4"
python -m vera.server.save_vis_video --vis-host localhost --vis-port "$VIS_PORT" \
  --output "$OUT_DIR/mimicgen_vis.mp4" --fps 10 || echo "[demo_closedloop_mimicgen] vis dump skipped"

echo "[demo_closedloop_mimicgen] server log tail:"; tail -n 15 "$OUT_DIR/mimicgen_server.log"
exit $RC
