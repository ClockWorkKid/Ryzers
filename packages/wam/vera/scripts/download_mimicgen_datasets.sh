#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Fetch MimicGen "core" task datasets (robosuite/MuJoCo hdf5) into the mounted model volume.
# These are the open-source MimicGen datasets hosted at HF `amandlek/mimicgen_datasets`
# (NVlabs MimicGen); we download them at runtime rather than re-hosting (rule 8). The hdf5
# carries the env config + demo initial states, so `run_mimicgen_eval --dataset <task>.hdf5`
# fully determines the task — the released VERA planner + taskbalanced Jacobian IDM cover all
# nine tasks below (coffee/square/stack/stack_three families).
#
#   ryzers run /ryzers/scripts/download_mimicgen_datasets.sh                 # all 9 tasks (~10-15 GB)
#   ryzers run /ryzers/scripts/download_mimicgen_datasets.sh stack_d0 square_d0   # a subset
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HF_COMMON:-$HERE/_hf_common.sh}"

# The 9 tasks the released MimicGen model was trained on (config_..._taskbalanced.yaml).
ALL_TASKS=(coffee_d0 coffee_d1 square_d0 square_d1 square_d2 stack_d0 stack_d1 stack_three_d0 stack_three_d1)
TASKS=("$@")
[ "${#TASKS[@]}" -gt 0 ] || TASKS=("${ALL_TASKS[@]}")

REPO="amandlek/mimicgen_datasets"
DEST="${MIMICGEN_DATASET_ROOT:-/models/mimicgen_datasets}"
mkdir -p "$DEST"

INCLUDES=()
for t in "${TASKS[@]}"; do INCLUDES+=(--include "core/${t}.hdf5"); done

echo "==> MimicGen core datasets -> $DEST : ${TASKS[*]}"
hf_prefetch "$REPO" --repo-type dataset --local-dir "$DEST" "${INCLUDES[@]}"

echo "PASS: MimicGen datasets cached under $DEST/core"
ls -la "$DEST/core" 2>/dev/null || true
