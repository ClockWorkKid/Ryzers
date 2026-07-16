#!/usr/bin/env bash
# Headless RandomPolicy rollout of a RoboLab-AMD task -> MP4 under $OUT_DIR (no model).
# Proves the ROCm/EGL render + MuJoCo step (JOINT_POSITION control) + video encode path.
# Fetches+converts the real YCB banana/bowl meshes on first run. Knobs: TASK, SEED, STEPS.
set -euo pipefail
bash /ryzers/scripts/setup_robolab.sh || true   # lazy fallback also handled in-env
exec python -m sim_robolab.sanity
