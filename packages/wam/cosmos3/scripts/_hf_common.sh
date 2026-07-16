#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Shared HF pre-fetch helper for the download_*.sh scripts. Source it, then call
# `hf_prefetch <repo> [args...]` (e.g. --include / --repo-type / --local-dir).
#
# The Cosmos3 weights (nvidia/Cosmos3-Nano-Policy-DROID) and dataset (nvidia/Cosmos3-DROID)
# are GATED — you must accept the OpenMDW license on HF and export HF_TOKEN. Defaults to a
# single stream + Xet off for reliable downloads, with retry/resume from cache.
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"

# Newer huggingface_hub ships the `hf` CLI; fall back to legacy `huggingface-cli`.
if command -v hf >/dev/null 2>&1; then HF_CLI="hf"; else HF_CLI="huggingface-cli"; fi
if [ -n "${HF_TOKEN:-}" ]; then
  "$HF_CLI" auth login --token "$HF_TOKEN" >/dev/null 2>&1 || \
    "$HF_CLI" login --token "$HF_TOKEN" >/dev/null 2>&1 || true
fi

hf_prefetch() {
  local tries="${DL_RETRIES:-5}" n=1
  echo "==> prefetch ($HF_CLI): $* (DISABLE_XET=$HF_HUB_DISABLE_XET HF_TRANSFER=$HF_HUB_ENABLE_HF_TRANSFER timeout=${HF_HUB_DOWNLOAD_TIMEOUT}s)"
  while true; do
    if "$HF_CLI" download "$@"; then echo "    cached: $1"; return 0; fi
    if [ "$n" -ge "$tries" ]; then
      echo "ERROR: '$HF_CLI download $*' failed after $tries attempts (gated repo? export HF_TOKEN + accept the license)" >&2
      return 1
    fi
    echo "    (attempt $n/$tries failed; retrying in 5s, resuming from cache...)" >&2
    n=$((n + 1)); sleep 5
  done
}
