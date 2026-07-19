#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Fetch MimicGen "core" task datasets (robosuite/MuJoCo hdf5) into the mounted assets volume.
# These are the open-source MimicGen datasets hosted at HF `amandlek/mimicgen_datasets`
# (NVlabs MimicGen); we download them at runtime rather than re-hosting (rule 8). The hdf5
# carries the env config + demo initial states, so `--dataset <task>.hdf5` fully determines
# the task — model-agnostic (any policy layered on this sim base is evaluated on the same task).
#
#   ryzers run --name sim-mimicgen /ryzers/scripts/download_mimicgen_datasets.sh                # all 9 (~10-15 GB)
#   ryzers run --name sim-mimicgen /ryzers/scripts/download_mimicgen_datasets.sh stack_d0 square_d0  # a subset
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HF_COMMON:-$HERE/_hf_common.sh}"

# The 9 MimicGen core tasks (coffee/square/stack/stack_three families).
ALL_TASKS=(coffee_d0 coffee_d1 square_d0 square_d1 square_d2 stack_d0 stack_d1 stack_three_d0 stack_three_d1)
TASKS=("$@")
[ "${#TASKS[@]}" -gt 0 ] || TASKS=("${ALL_TASKS[@]}")

REPO="amandlek/mimicgen_datasets"
DEST="${MIMICGEN_DATASET_ROOT:-/sim_data/mimicgen_datasets}"
mkdir -p "$DEST"

INCLUDES=()
for t in "${TASKS[@]}"; do INCLUDES+=(--include "core/${t}.hdf5"); done

echo "==> MimicGen core datasets -> $DEST : ${TASKS[*]}"
hf_prefetch "$REPO" --repo-type dataset --local-dir "$DEST" "${INCLUDES[@]}"

echo "PASS: MimicGen datasets cached under $DEST/core"
ls -la "$DEST/core" 2>/dev/null || true
