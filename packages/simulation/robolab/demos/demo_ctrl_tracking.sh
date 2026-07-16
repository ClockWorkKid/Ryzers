#!/usr/bin/env bash
# Gate G2: validate JOINT_POSITION controller tracking on the real task scene (no model).
# Runs the scripted joint trajectory, writes a commanded-vs-achieved overlay + motion MP4 to
# $OUT_DIR, and prints per-joint RMSE + PASS/FAIL. Knobs: TASK, SEED, STEPS, AMPLITUDE, PERIOD.
set -euo pipefail
bash /ryzers/scripts/setup_robolab.sh || true   # ensure real YCB assets present
exec python -m sim_robolab.tracking
