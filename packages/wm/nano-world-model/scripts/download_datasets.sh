#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Fetch the dataset a rollout uses for its history/context frames. Sources are the upstream
# canonical ones (nothing re-hosted here, rule 8); see upstream docs/datasets/README.md.
#   - DINO-WM (point_maze/pusht/wall/rope/granular): DINO-WM OSF project (osf.io/bmw48).
#   - RT-1 fractal / CSGO: HuggingFace (large; fetch explicitly with DOMAIN=rt1|csgo).
#
# OSF waterbutler file ids verified against api.osf.io/v2/nodes/bmw48 (datasets/ folder).
set -euo pipefail

DOMAIN="${DOMAIN:-pusht}"
DATASET_DIR="${DATASET_DIR:-/models/datasets/dino_wm}"
OSF_VIEW="${OSF_VIEW:-a56a296ce3b24cceaf408383a175ce28}"   # DINO-WM public view-only token
WB="https://files.osf.io/v1/resources/bmw48/providers/osfstorage"

# DINO-WM zip -> (osf file id, extracted dir marker).
declare -A OSF_ID=(
  [pusht]=678ac7164c237c16bedd6129
  [point_maze]=678ac918567274d368282c6d
  [wall]=678ac8fff57516fd9edd6046 )
declare -A ZIPNAME=(
  [pusht]=pusht_noise.zip [point_maze]=point_maze.zip [wall]=wall_single.zip )
declare -A OUTDIR=(
  [pusht]=pusht_noise [point_maze]=point_maze [wall]=wall_single )

osf_get() {  # $1=id $2=outfile
  [ -f "$2" ] && { echo "have $2"; return; }
  curl -L --fail --retry 3 -o "$2" "${WB}/$1?view_only=${OSF_VIEW}"
}

case "$DOMAIN" in
  rt1)
    : "${RT1_DATA_ROOT:=/models/datasets/rt1_fractal}"
    mkdir -p "$RT1_DATA_ROOT"
    # SAFETY: the full LeRobot fractal set (87k eps) is >100 GB. Never pull it implicitly.
    # The rollout's LeRobotDataSource fetches ONLY the episodes it is given (episodes=range(n_rollout))
    # on-demand into RT1_DATA_ROOT, so a bounded dataset.loader.n_rollout keeps disk use small and we
    # deliberately DO NOT bulk-download here. Set RT1_INCLUDE only to pre-seed a manual mirror.
    if [ -z "${RT1_INCLUDE:-}" ]; then
      echo "[rt1] No bulk download: lerobot will fetch a bounded episode subset on-demand"
      echo "      (dataset.loader.n_rollout) into $RT1_DATA_ROOT. Set RT1_INCLUDE to pre-seed a mirror."
      exit 0
    fi
    hf download IPEC-COMMUNITY/fractal20220817_data_lerobot --repo-type dataset \
      --local-dir "$RT1_DATA_ROOT" --include "$RT1_INCLUDE"
    ;;
  csgo)
    : "${CSGO_DATA_DIR:=/models/datasets/csgo}"
    mkdir -p "$CSGO_DATA_DIR"
    # Idempotent: the CSGO data source reads files from <data_dir>/<start>-<end>/hdf5_dm_july2021_N.hdf5.
    # If any extracted shard is already laid out, we're done.
    if ls "$CSGO_DATA_DIR"/*/*.hdf5 >/dev/null 2>&1; then
      echo "[csgo] have extracted subset under $CSGO_DATA_DIR (skipping download)"; exit 0
    fi
    # SAFETY: the full CSGO set is ~675 GB (37 shards × ~18 GB). Never pull it implicitly —
    # require an explicit CSGO_INCLUDE glob of 200-episode shard tar(s). Default to the first shard,
    # which contains the lowest-numbered validation episodes (test_split head: 3,5,19,27,...).
    CSGO_INCLUDE="${CSGO_INCLUDE:-hdf5_dm_july2021_1_to_200.tar}"
    echo "[csgo] downloading shard(s): $CSGO_INCLUDE"
    hf download teapearce/CounterStrike_Deathmatch --repo-type dataset \
      --local-dir "$CSGO_DATA_DIR" --include "$CSGO_INCLUDE"
    # Extract each shard tar and arrange its .hdf5 files into the <start>-<end>/ layout by episode number.
    for tar in "$CSGO_DATA_DIR"/hdf5_dm_july2021_*.tar; do
      [ -f "$tar" ] || continue
      echo "[csgo] extracting $(basename "$tar")"
      rm -rf "$CSGO_DATA_DIR/_extract"; mkdir -p "$CSGO_DATA_DIR/_extract"
      tar xf "$tar" -C "$CSGO_DATA_DIR/_extract"
      find "$CSGO_DATA_DIR/_extract" -name "hdf5_dm_july2021_*.hdf5" | while read -r f; do
        n=$(basename "$f" .hdf5); n=${n##*_}
        start=$(( ((n - 1) / 200) * 200 + 1 )); end=$(( start + 199 ))
        mkdir -p "$CSGO_DATA_DIR/${start}-${end}"
        mv -n "$f" "$CSGO_DATA_DIR/${start}-${end}/"
      done
      rm -f "$tar"
    done
    rm -rf "$CSGO_DATA_DIR/_extract"
    echo "[csgo] arranged $(ls "$CSGO_DATA_DIR"/*/*.hdf5 2>/dev/null | wc -l) hdf5 files"
    ;;
  rope|granular)
    # rope + granular share the deformable/ split archive (~15 GB). Needs `zip`+`unzip`.
    mkdir -p "$DATASET_DIR"; cd "$DATASET_DIR"
    if [ -d deformable/"$DOMAIN" ]; then echo "have deformable/$DOMAIN"; exit 0; fi
    osf_get 678ad9d568ad6a43dfdd6006 deformable.zip
    osf_get 6793d46b1b745202d8b537d2 deformable.z01
    osf_get 6793d44f2db8ab2633a49349 deformable.z02
    osf_get 6793d3c89015e736d3df3b98 deformable.z03
    echo "recombining split archive..."
    zip -s 0 deformable.zip --out deformable_full.zip
    unzip -o -q deformable_full.zip
    echo "extracted -> $DATASET_DIR/deformable"
    ;;
  *)
    id="${OSF_ID[$DOMAIN]:-}"
    [ -n "$id" ] || { echo "Unknown DOMAIN=$DOMAIN (dino_wm: pusht point_maze wall rope granular; or rt1/csgo)"; exit 1; }
    mkdir -p "$DATASET_DIR"; cd "$DATASET_DIR"
    out="${OUTDIR[$DOMAIN]}"
    [ -d "$out" ] && { echo "have $DATASET_DIR/$out"; exit 0; }
    osf_get "$id" "${ZIPNAME[$DOMAIN]}"
    python3 -m zipfile -e "${ZIPNAME[$DOMAIN]}" .
    echo "Extracted -> $DATASET_DIR/$out (expect $out/{train,val})"
    ;;
esac
