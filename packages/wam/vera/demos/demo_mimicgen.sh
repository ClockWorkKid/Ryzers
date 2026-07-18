#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# MimicGen closed-loop demo (runs INSIDE the vera container). robosuite/MuJoCo 2-block stack
# (stack_d0) driven by the omni WAN-1.3B video planner + mimicgen Jacobian IDM, all inference
# server-side. Uses the UPSTREAM controller `vera.controller.run_mimicgen_eval` unchanged
# (rule 2.1); it sets MUJOCO_GL=egl itself. Offscreen render on gfx1151 via EGL, osmesa fallback.
#   1) start_vera_server --embodiment mimicgen --algo-config <omni> --sample-steps 10  (bg; vis @ --vis-port)
#   2) wait for the policy port
#   3) run_mimicgen_eval --dataset stack_d0.hdf5 ...  (fg; EGL, retry osmesa on GL failure)
#   4) save_vis_video -> $OUT_DIR/mimicgen_vis.mp4  (dream|tracks|jacobian composite)
# Weights: download_checkpoints.sh wave1 (mimicgen-wan-1.3b + idm-mimicgen) + Wan2.1-T2V-1.3B base.
#   ryzers run /ryzers/demos/demo_mimicgen.sh
set -uo pipefail

: "${PORT:=8800}"
: "${VIS_PORT:=8801}"
# TASK selects the MimicGen dataset; the released planner + taskbalanced IDM cover the 9 tasks:
#   coffee_d0 coffee_d1 square_d0 square_d1 square_d2 stack_d0 stack_d1 stack_three_d0 stack_three_d1
# Fetch them with scripts/download_mimicgen_datasets.sh (rule 8).
: "${TASK:=stack_d0}"
: "${DATASET:=/models/mimicgen_datasets/core/${TASK}.hdf5}"
: "${OUT_DIR:=/outputs/vera_mimicgen/${TASK}}"
: "${NUM_DEMOS:=3}"
: "${ROLLOUT_HORIZON:=400}"
: "${RENDER_SIZE:=128}"
: "${SAMPLE_STEPS:=10}"
# WAN base (VAE/T5/base DiT dir) + MimicGen-specialist bundle (tuned DiT + flow_decoder + algo_config) + IDM.
: "${VERA_WAN_CKPT_ROOT:=/models/wan2.1-t2v-1.3b}"
: "${VERA_MIMICGEN_CKPT_DIR:=/models/vera-ckpts/mimicgen-wan-1.3b}"
: "${VERA_MIMICGEN_DYNAMICS_CKPT:=/models/vera-ckpts/idm-mimicgen-285ouq1q/model.ckpt}"
ALGO_CONFIG="${ALGO_CONFIG:-$VERA_MIMICGEN_CKPT_DIR/algo_config.yaml}"
export VERA_WAN_CKPT_ROOT VERA_MIMICGEN_CKPT_DIR VERA_MIMICGEN_DYNAMICS_CKPT
mkdir -p "$OUT_DIR"

if [ ! -f "$DATASET" ]; then
  echo "[demo_mimicgen] dataset not found: $DATASET"
  echo "[demo_mimicgen] fetch it first: ryzers run /ryzers/scripts/download_mimicgen_datasets.sh $TASK"
  exit 1
fi
echo "[demo_mimicgen] TASK=$TASK DATASET=$DATASET"

# Motion-tracker backend: the shipped algo_config has no `tracker:` block, so the WAN pipeline's
# MotionTrackConfig defaults to backend="alltracker". alltracker is a vendored-only checkpoint
# (not pip/torch.hub installable) and — per VERA's own build_policy note — produces wrong-direction
# flow on MimicGen ("arm flees blocks"). cotracker is the intended MimicGen tracker: CoTrackerInference
# pulls facebookresearch/co-tracker via torch.hub at runtime (rule 8), no baked binary. We layer a
# `tracker.backend: cotracker` override onto a COPY of the config (never edit the downloaded asset);
# the patched tracker_backend_from_cfg (build step) makes MotionTrackConfig.backend actually take effect.
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

echo "[demo_mimicgen] WAN_ROOT=$VERA_WAN_CKPT_ROOT MG_DIR=$VERA_MIMICGEN_CKPT_DIR"
echo "[demo_mimicgen] tracker backend = cotracker (algo_config: $ALGO_CONFIG)"
echo "[demo_mimicgen] starting policy server (port=$PORT vis=$VIS_PORT steps=$SAMPLE_STEPS)"
python -m vera.server.start_vera_server --embodiment mimicgen --port "$PORT" --vis-port "$VIS_PORT" \
  --algo-config "$ALGO_CONFIG" --sample-steps "$SAMPLE_STEPS" \
  > "$OUT_DIR/mimicgen_server.log" 2>&1 &
SRV=$!
cleanup() { echo "[demo_mimicgen] stopping server ($SRV)"; kill "$SRV" 2>/dev/null || true; }
trap cleanup EXIT

echo "[demo_mimicgen] waiting for policy port $PORT (WAN+IDM load is slow, ~minutes) ..."
for i in $(seq 1 200); do
  if ! kill -0 "$SRV" 2>/dev/null; then echo "[demo_mimicgen] server died early:"; tail -n 60 "$OUT_DIR/mimicgen_server.log"; exit 1; fi
  if python -c "import socket,sys; s=socket.socket(); s.settimeout(2); sys.exit(0 if s.connect_ex(('127.0.0.1',$PORT))==0 else 1)"; then
    echo "[demo_mimicgen] policy port up"; break
  fi
  sleep 5
done

run_client() {  # $1 = GL backend
  echo "[demo_mimicgen] rollout with MUJOCO_GL=$1"
  MUJOCO_GL="$1" PYOPENGL_PLATFORM="$1" python -u -m vera.controller.run_mimicgen_eval \
    --host 127.0.0.1 --port "$PORT" --dataset "$DATASET" \
    --num-demos "$NUM_DEMOS" --rollout-horizon "$ROLLOUT_HORIZON" --render-size "$RENDER_SIZE" \
    --output-dir "$OUT_DIR"
}
run_client egl; RC=$?
if [ "$RC" -ne 0 ]; then
  echo "[demo_mimicgen] EGL path failed (rc=$RC); retrying with software osmesa"
  run_client osmesa; RC=$?
fi

echo "[demo_mimicgen] dumping viewer buffer -> $OUT_DIR/mimicgen_vis.mp4"
python -m vera.server.save_vis_video --vis-host localhost --vis-port "$VIS_PORT" \
  --output "$OUT_DIR/mimicgen_vis.mp4" --fps 10 || echo "[demo_mimicgen] vis dump skipped"

echo "[demo_mimicgen] server log tail:"; tail -n 15 "$OUT_DIR/mimicgen_server.log"
exit $RC
