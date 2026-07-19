#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# MimicGen FULL-RANGE closed-loop suite (runs INSIDE the vera container). Starts ONE policy
# server (WAN-1.3B planner + taskbalanced Jacobian IDM) and evaluates it across every task the
# released model covers, reusing the warm server so the model loads once (cold start ~340s, then
# ~warm per chunk). Task is selected purely by the dataset hdf5 (rule 2.1: upstream
# run_mimicgen_eval unchanged); MuJoCo renders offscreen via EGL, osmesa fallback per task.
#   TASKS default = the 9 trained tasks; override with args or the TASKS env.
#   NUM_DEMOS/ROLLOUT_HORIZON default to a short validation rollout; raise for a full eval.
#
#   ryzers run /ryzers/demos/demo_mimicgen_suite.sh                          # all 9, short
#   TASKS="square_d0 coffee_d0" NUM_DEMOS=3 ROLLOUT_HORIZON=200 ryzers run /ryzers/demos/demo_mimicgen_suite.sh
set -uo pipefail

: "${PORT:=8800}"
: "${VIS_PORT:=8801}"
: "${SAMPLE_STEPS:=10}"
: "${NUM_DEMOS:=1}"
: "${ROLLOUT_HORIZON:=100}"
: "${RENDER_SIZE:=128}"
: "${OUT_ROOT:=/outputs/vera_mimicgen_suite}"
# Datasets are fetched by the simulation/mimicgen base into its /sim_data mount (rule 8).
: "${MIMICGEN_DATASET_ROOT:=/sim_data/mimicgen_datasets}"
: "${DATASET_ROOT:=$MIMICGEN_DATASET_ROOT/core}"
: "${VERA_WAN_CKPT_ROOT:=/models/wan2.1-t2v-1.3b}"
: "${VERA_MIMICGEN_CKPT_DIR:=/models/vera-ckpts/mimicgen-wan-1.3b}"
: "${VERA_MIMICGEN_DYNAMICS_CKPT:=/models/vera-ckpts/idm-mimicgen-285ouq1q/model.ckpt}"
export VERA_WAN_CKPT_ROOT VERA_MIMICGEN_CKPT_DIR VERA_MIMICGEN_DYNAMICS_CKPT

DEFAULT_TASKS="coffee_d0 coffee_d1 square_d0 square_d1 square_d2 stack_d0 stack_d1 stack_three_d0 stack_three_d1"
if [ "$#" -gt 0 ]; then TASKS="$*"; else TASKS="${TASKS:-$DEFAULT_TASKS}"; fi
mkdir -p "$OUT_ROOT"
SUMMARY="$OUT_ROOT/suite_summary.tsv"
: > "$SUMMARY"

# cotracker backend override on a COPY of the shipped algo_config (see demo_mimicgen.sh notes).
ALGO_CONFIG="${ALGO_CONFIG:-$VERA_MIMICGEN_CKPT_DIR/algo_config.yaml}"
PATCHED_ALGO="$OUT_ROOT/algo_config.cotracker.yaml"
cp "$ALGO_CONFIG" "$PATCHED_ALGO"
if ! grep -qE '^tracker:' "$PATCHED_ALGO"; then
  { echo ""; echo "tracker:"; echo "  backend: cotracker"; echo "  enabled: true"; echo "  return_visualization: true"; } >> "$PATCHED_ALGO"
fi
ALGO_CONFIG="$PATCHED_ALGO"

echo "[suite] tasks: $TASKS"
echo "[suite] starting policy server (port=$PORT vis=$VIS_PORT steps=$SAMPLE_STEPS)"
python -m vera.server.start_vera_server --embodiment mimicgen --port "$PORT" --vis-port "$VIS_PORT" \
  --algo-config "$ALGO_CONFIG" --sample-steps "$SAMPLE_STEPS" \
  > "$OUT_ROOT/server.log" 2>&1 &
SRV=$!
cleanup() { echo "[suite] stopping server ($SRV)"; kill "$SRV" 2>/dev/null || true; }
trap cleanup EXIT

echo "[suite] waiting for policy port $PORT (WAN+IDM load is slow, ~minutes) ..."
for i in $(seq 1 200); do
  if ! kill -0 "$SRV" 2>/dev/null; then echo "[suite] server died early:"; tail -n 60 "$OUT_ROOT/server.log"; exit 1; fi
  if python -c "import socket,sys; s=socket.socket(); s.settimeout(2); sys.exit(0 if s.connect_ex(('127.0.0.1',$PORT))==0 else 1)"; then
    echo "[suite] policy port up"; break
  fi
  sleep 5
done

run_task_gl() {  # $1 task, $2 GL backend
  local task="$1" gl="$2" out="$OUT_ROOT/$1"
  mkdir -p "$out"
  MUJOCO_GL="$gl" PYOPENGL_PLATFORM="$gl" python -u -m vera.controller.run_mimicgen_eval \
    --host 127.0.0.1 --port "$PORT" --dataset "$DATASET_ROOT/$task.hdf5" \
    --num-demos "$NUM_DEMOS" --rollout-horizon "$ROLLOUT_HORIZON" --render-size "$RENDER_SIZE" \
    --output-dir "$out" 2>&1 | tee "$out/eval.log"
  return "${PIPESTATUS[0]}"
}

for task in $TASKS; do
  ds="$DATASET_ROOT/$task.hdf5"
  if [ ! -f "$ds" ]; then
    echo "[suite] SKIP $task (missing $ds — run download_mimicgen_datasets.sh $task)"
    printf "%s\tMISSING_DATASET\n" "$task" >> "$SUMMARY"; continue
  fi
  echo "======================================================================"
  echo "[suite] TASK=$task  (demos=$NUM_DEMOS horizon=$ROLLOUT_HORIZON)"
  run_task_gl "$task" egl; rc=$?
  if [ "$rc" -ne 0 ]; then echo "[suite] $task EGL failed (rc=$rc); retry osmesa"; run_task_gl "$task" osmesa; rc=$?; fi
  # snapshot the viewer buffer for this task
  python -m vera.server.save_vis_video --vis-host localhost --vis-port "$VIS_PORT" \
    --output "$OUT_ROOT/$task/${task}_vis.mp4" --fps 10 || echo "[suite] $task vis dump skipped"
  sr="$(grep -oE 'success rate:[^%]*%' "$OUT_ROOT/$task/eval.log" 2>/dev/null | tail -1)"
  printf "%s\trc=%s\t%s\n" "$task" "$rc" "${sr:-n/a}" >> "$SUMMARY"
  echo "[suite] $task done (rc=$rc) ${sr:-}"
done

echo "======================================================================"
echo "[suite] SUMMARY ($SUMMARY):"; cat "$SUMMARY"
