#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Fetch RoboCasa's ~10GB kitchen assets (objects, fixtures, textures) into the mounted
# volume and wire them into the editable robocasa install at /opt/robocasa, so nothing
# large is baked into the image (rules 3 & 8). Idempotent: re-run is a no-op once the
# marker exists.
#   Assets: robocasa.scripts.download_kitchen_assets (upstream downloader)
set -euo pipefail

ASSETS="${ROBOCASA_ASSETS_DIR:-/models/robocasa_assets}"
# find_spec resolves the package dir WITHOUT executing robocasa/__init__.py (which prints a
# "mimicgen not installed" warning to stdout that would otherwise corrupt this substitution).
ROBOCASA_PKG="$(python -c 'import importlib.util, os; print(os.path.dirname(importlib.util.find_spec("robocasa").origin))')"
PKG_ASSETS="$ROBOCASA_PKG/models/assets"
mkdir -p "$ASSETS"

# Redirect the package asset dir onto the mounted volume. On first run, seed the volume
# with whatever small assets ship in the repo, then replace the package dir with a symlink
# so the big downloaded packs land on the volume.
if [ ! -L "$PKG_ASSETS" ]; then
  if [ -d "$PKG_ASSETS" ] && [ ! -e "$ASSETS/.seeded" ]; then
    echo "== seed volume from packaged assets =="
    cp -an "$PKG_ASSETS/." "$ASSETS/" 2>/dev/null || true
    touch "$ASSETS/.seeded"
  fi
  rm -rf "$PKG_ASSETS"
  ln -sfn "$ASSETS" "$PKG_ASSETS"
  echo "  linked $PKG_ASSETS -> $ASSETS"
fi

# Consider assets present if the extracted packs exist, even when the .ready marker was
# lost (e.g. a wiped tmp/marker or an interrupted prior run) -- avoids a needless ~4GB
# re-download. Backfill the marker in that case so later runs short-circuit on it too.
if [ ! -f "$ASSETS/.ready" ] \
   && [ -d "$ASSETS/fixtures" ] && [ -d "$ASSETS/objects" ] && [ -d "$ASSETS/textures" ]; then
  echo "== assets already extracted (fixtures/objects/textures present); backfilling marker =="
  touch "$ASSETS/.ready" 2>/dev/null || true
fi

if [ ! -f "$ASSETS/.ready" ]; then
  echo "== download RoboCasa kitchen assets (~8GB) into $ASSETS =="
  # The downloader writes into robocasa/models/assets (now the symlinked volume) and prompts
  # once for confirmation. Feed 'y' via a heredoc (NOT `yes |`: under `set -o pipefail` the
  # SIGPIPE that `yes` gets when the reader exits would fail the pipeline). `|| true` so a
  # transient network hiccup doesn't abort; success is verified by the extracted dirs below.
  python -m robocasa.scripts.download_kitchen_assets <<'EOF' || true
y
y
y
y
y
y
EOF
  if [ -d "$ASSETS/fixtures" ] && [ -d "$ASSETS/objects" ] && [ -d "$ASSETS/textures" ]; then
    touch "$ASSETS/.ready"
  else
    echo "WARNING: RoboCasa asset download looks incomplete (missing fixtures/objects/textures)" >&2
  fi
fi

echo "PASS: RoboCasa assets ready under $ASSETS"
