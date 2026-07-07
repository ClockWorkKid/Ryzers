#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# (Re)download the LIBERO-Plus perturbation asset pack (~6.4 GB: hundreds of objects,
# textures, scenes) from the upstream open-source HuggingFace dataset and unzip it into the
# installed LIBERO-Plus assets tree. The base image already bakes these at build time; this
# script exists as an explicit, idempotent download option (refresh, or for a build done
# with the asset layer skipped). Never re-hosts the binaries here (compliance).
#
#   ryzers run --name sim-libero-plus /ryzers/demos/demo_download.sh
set -euo pipefail

LIBERO_ROOT="${LIBERO_ROOT:-/opt/LIBERO-plus}"
ASSETS_DIR="${LIBERO_ROOT}/libero/libero/assets"
REPO_ID="${LIBEROPLUS_HF_REPO:-Sylvest/LIBERO-plus}"

export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DISABLE_TELEMETRY=1

if [ -d "${ASSETS_DIR}/textures" ] && [ -d "${ASSETS_DIR}/new_objects" ] && [ "${FORCE:-0}" != "1" ]; then
  echo "[download] assets already present at ${ASSETS_DIR} (set FORCE=1 to re-download)."
  exit 0
fi

echo "[download] fetching assets.zip from HuggingFace dataset ${REPO_ID} ..."
tmp="$(mktemp -d)"
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='${REPO_ID}', repo_type='dataset', filename='assets.zip', local_dir='${tmp}')"

echo "[download] unzipping into ${LIBERO_ROOT}/libero/libero ..."
mkdir -p "${LIBERO_ROOT}/libero/libero"
unzip -q -o "${tmp}/assets.zip" -d "${LIBERO_ROOT}/libero/libero"
rm -rf "${tmp}"

echo "[download] done. assets at ${ASSETS_DIR}:"
ls -1 "${ASSETS_DIR}" | sed 's/^/  /'
