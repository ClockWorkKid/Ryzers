#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Shared HF pre-fetch helper for the download_*.sh scripts. Source it, then call
# `hf_prefetch <repo> [--repo-type dataset] [--include PATTERN] --local-dir DIR`.
#
# Uses the modern `hf` CLI (huggingface_hub>=0.25, as upstream X-WAM's evaluation/README
# does). Defaults to a single stream + Xet client off for reliable anonymous downloads, and
# retries so a dropped connection resumes from cache. Set HF_HUB_ENABLE_HF_TRANSFER=1 for
# parallel speed where permits/auth allow.
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"

HF_BIN="hf"
command -v hf >/dev/null 2>&1 || HF_BIN="huggingface-cli"

if [ -n "${HF_TOKEN:-}" ]; then
  "$HF_BIN" auth login --token "$HF_TOKEN" >/dev/null 2>&1 || \
  "$HF_BIN" login --token "$HF_TOKEN" >/dev/null 2>&1 || true
fi

hf_prefetch() {
  local tries="${DL_RETRIES:-5}" n=1
  echo "==> prefetch: $* (DISABLE_XET=$HF_HUB_DISABLE_XET HF_TRANSFER=$HF_HUB_ENABLE_HF_TRANSFER timeout=${HF_HUB_DOWNLOAD_TIMEOUT}s)"
  while true; do
    if "$HF_BIN" download "$@"; then echo "    cached: $1"; return 0; fi
    if [ "$n" -ge "$tries" ]; then
      echo "ERROR: '$HF_BIN download $*' failed after $tries attempts" >&2
      return 1
    fi
    echo "    (attempt $n/$tries failed; retrying in 5s, resuming from cache...)" >&2
    n=$((n + 1)); sleep 5
  done
}
