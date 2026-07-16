#!/usr/bin/env bash
# Gate G3: validate the object_in_container(banana, bowl) predicate on the real YCB assets.
# Drives the real banana/bowl into 1 positive + 3 negative configurations (genuine gripper
# contact + physics), prints per-scenario predicate booleans + PASS/FAIL, and writes an
# annotated 4-panel image to $OUT_DIR. Knobs: TASK, SEED.
set -euo pipefail
bash /ryzers/scripts/setup_robolab.sh || true   # ensure real YCB assets present
exec python -m sim_robolab.predicate_check
