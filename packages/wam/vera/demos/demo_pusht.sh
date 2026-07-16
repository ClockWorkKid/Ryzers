#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# PushT closed-loop demo (runs INSIDE the vera container). Brings up the full VERA closed-loop
# stack for the light PushT embodiment (DFoT planner + Jacobian IDM, no MuJoCo):
#   1) start_vera_server --embodiment pusht  (background; MJPEG viewer on --vis-port)
#   2) wait for the policy port to accept connections
#   3) closedloop_pusht.py  (foreground; steps the gym-pusht env, inference server-side)
#   4) save_vis_video -> $OUT_DIR/pusht_vis.mp4  (dump the viewer buffer: dream|tracks|jacobian)
# Weights: download_checkpoints.sh wave1  (pusht-dfot + pusht-idm) + the PushT replay zarr.
#   ryzers run /ryzers/demos/demo_pusht.sh
# Optional env: PORT VIS_PORT ZARR_PATH OUT_DIR FRAME_INDICES N_REPEATS HORIZON SEED
set -uo pipefail

: "${PORT:=8820}"
: "${VIS_PORT:=8821}"
: "${ZARR_PATH:=/models/pusht/pusht_cchi_v7_replay.zarr}"
: "${OUT_DIR:=/outputs/vera_pusht}"
: "${VERA_PUSHT_PLANNER_CKPT:=/models/vera-ckpts/pusht-dfot/model.ckpt}"
: "${VERA_PUSHT_DYNAMICS_CKPT:=/models/vera-ckpts/pusht-idm/model.ckpt}"
export VERA_PUSHT_PLANNER_CKPT VERA_PUSHT_DYNAMICS_CKPT VERA_HOST=127.0.0.1 VERA_PORT="$PORT" VIS_PORT ZARR_PATH OUT_DIR
mkdir -p "$OUT_DIR"

echo "[demo_pusht] starting policy server (port=$PORT vis=$VIS_PORT)"
python -m vera.server.start_vera_server --embodiment pusht --port "$PORT" --vis-port "$VIS_PORT" \
  > "$OUT_DIR/pusht_server.log" 2>&1 &
SRV=$!
cleanup() { echo "[demo_pusht] stopping server ($SRV)"; kill "$SRV" 2>/dev/null || true; }
trap cleanup EXIT

echo "[demo_pusht] waiting for policy port $PORT ..."
for i in $(seq 1 120); do
  if ! kill -0 "$SRV" 2>/dev/null; then echo "[demo_pusht] server died early:"; tail -n 40 "$OUT_DIR/pusht_server.log"; exit 1; fi
  if python -c "import socket,sys; s=socket.socket(); s.settimeout(2); sys.exit(0 if s.connect_ex(('127.0.0.1',$PORT))==0 else 1)"; then
    echo "[demo_pusht] policy port up after ${i}0s-ish"; break
  fi
  sleep 3
done

echo "[demo_pusht] running closed-loop client"
python -u /ryzers/scripts/closedloop_pusht.py
RC=$?

echo "[demo_pusht] dumping viewer buffer -> $OUT_DIR/pusht_vis.mp4"
python -m vera.server.save_vis_video --vis-host localhost --vis-port "$VIS_PORT" \
  --output "$OUT_DIR/pusht_vis.mp4" --fps 10 || echo "[demo_pusht] vis dump skipped (no buffer / recorder error)"

echo "[demo_pusht] server log tail:"; tail -n 15 "$OUT_DIR/pusht_server.log"
exit $RC
