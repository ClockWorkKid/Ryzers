#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Shared HF pre-fetch helper for the FlowWAM download_*.sh scripts. Source it, then call
# `hf_prefetch <repo> [files...] [--local-dir DIR] [--repo-type dataset]`.
# Uses the modern `hf` CLI (huggingface-hub>=0.34, installed in the image), single stream +
# Xet off for reliable downloads, with retry/resume. Logs in with $HF_TOKEN when present
# (needed for gated repos / the RoboTwin embodiments dataset).
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"

if [ -n "${HF_TOKEN:-}" ]; then
  hf auth login --token "$HF_TOKEN" >/dev/null 2>&1 || \
    huggingface-cli login --token "$HF_TOKEN" >/dev/null 2>&1 || true
fi

_hf_bin() { command -v hf >/dev/null 2>&1 && echo hf || echo huggingface-cli; }

hf_prefetch() {
  local bin; bin="$(_hf_bin)"
  local tries="${DL_RETRIES:-5}" n=1
  echo "==> prefetch ($bin): $* (DISABLE_XET=$HF_HUB_DISABLE_XET timeout=${HF_HUB_DOWNLOAD_TIMEOUT}s)"
  while true; do
    if "$bin" download "$@"; then echo "    cached: $1"; return 0; fi
    if [ "$n" -ge "$tries" ]; then
      echo "ERROR: '$bin download $*' failed after $tries attempts" >&2
      return 1
    fi
    echo "    (attempt $n/$tries failed; retrying in 5s, resuming from cache...)" >&2
    n=$((n + 1)); sleep 5
  done
}
