#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Fetch + unpack the AI2-THOR Unity CloudRendering build from the allenai public bucket into the
# ai2thor cache (~/.ai2thor/releases/<build>). Nothing is re-hosted (rule 8): the URL is derived
# from the COMMIT_ID baked into the installed ai2thor package. Safe to run at Docker build time
# (no GPU needed to download/extract) and at runtime into the mounted cache. Idempotent.
set -uo pipefail

PLATFORM="${THOR_PLATFORM:-CloudRendering}"
BASE_URL="https://ai2-thor-public.allenai.org/builds"
CACHE="${HOME:-/root}/.ai2thor/releases"

# CloudRendering has its OWN build commit (DEFAULT_CLOUDRENDERING_COMMIT_ID), distinct from the
# package-wide COMMIT_ID used by the Linux64/OSX builds. Using the wrong one 404s (the URL points at
# a nonexistent build). Resolve per-platform so build-time prefetch matches exactly what the ai2thor
# Controller fetches at runtime.
COMMIT_ID="$(PLATFORM="$PLATFORM" python - <<'PY'
import os
platform = os.environ.get("PLATFORM", "CloudRendering")
def commit_id():
    try:
        import ai2thor.build as b
        if platform == "CloudRendering":
            cid = getattr(b, "DEFAULT_CLOUDRENDERING_COMMIT_ID", None)
            if cid:
                return cid
        return getattr(b, "COMMIT_ID", "")
    except Exception:
        pass
    try:
        import ai2thor._builds as b
        return getattr(b, "COMMIT_ID", "")
    except Exception:
        return ""
print(commit_id() or "")
PY
)"

if [ -z "$COMMIT_ID" ]; then
  echo "[prefetch_build] could not determine ai2thor COMMIT_ID; will let Controller fetch at runtime"
  exit 1
fi

BUILD="thor-${PLATFORM}-${COMMIT_ID}"
DEST="${CACHE}/${BUILD}"
BIN="${DEST}/${BUILD}"

if [ -x "$BIN" ]; then
  echo "[prefetch_build] already present: $BIN"
  exit 0
fi

mkdir -p "$CACHE"
TMP="$(mktemp -d)"
URL="${BASE_URL}/${BUILD}.zip"
echo "[prefetch_build] downloading $URL"
if command -v wget >/dev/null 2>&1; then
  wget -q -O "${TMP}/${BUILD}.zip" "$URL" || { echo "[prefetch_build] download failed"; rm -rf "$TMP"; exit 1; }
else
  curl -fL -o "${TMP}/${BUILD}.zip" "$URL" || { echo "[prefetch_build] download failed"; rm -rf "$TMP"; exit 1; }
fi

echo "[prefetch_build] extracting -> $DEST"
mkdir -p "$DEST"
unzip -q -o "${TMP}/${BUILD}.zip" -d "$CACHE" || { echo "[prefetch_build] unzip failed"; rm -rf "$TMP"; exit 1; }
rm -rf "$TMP"

# ai2thor expects the Unity player executable to be +x.
[ -f "$BIN" ] && chmod +x "$BIN"
# The vulkan/plugin shared objects sometimes ship without +x; harmless to chmod broadly.
find "$DEST" -name "*.so*" -exec chmod +x {} + 2>/dev/null || true

if [ -x "$BIN" ]; then
  echo "[prefetch_build] OK: $BIN"
else
  echo "[prefetch_build] WARN: expected binary not found at $BIN (layout drift); runtime will re-fetch"
  exit 1
fi
