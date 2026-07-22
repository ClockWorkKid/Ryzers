#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Shared HF pre-fetch helper for the ImageWAM download_*.sh scripts. Source it, then call
# `hf_prefetch <repo> [files...] [--local-dir DIR] [--repo-type model|dataset]`.
# Defaults to a single stream + Xet off for reliable downloads, and retries so a dropped
# connection resumes from cache. Gated FLUX.2 repos need HF_TOKEN with granted access.
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"

if [ -n "${HF_TOKEN:-}" ]; then
  huggingface-cli login --token "$HF_TOKEN" >/dev/null 2>&1 || true
fi

hf_prefetch() {
  local tries="${DL_RETRIES:-5}" n=1
  echo "==> prefetch: $* (DISABLE_XET=$HF_HUB_DISABLE_XET HF_TRANSFER=$HF_HUB_ENABLE_HF_TRANSFER timeout=${HF_HUB_DOWNLOAD_TIMEOUT}s)"
  while true; do
    if huggingface-cli download "$@"; then echo "    cached: $1"; return 0; fi
    if [ "$n" -ge "$tries" ]; then
      echo "ERROR: 'huggingface-cli download $*' failed after $tries attempts" >&2
      return 1
    fi
    echo "    (attempt $n/$tries failed; retrying in 5s, resuming from cache...)" >&2
    n=$((n + 1)); sleep 5
  done
}
