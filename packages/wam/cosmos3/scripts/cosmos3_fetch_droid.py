# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Fetch a small slice of the gated nvidia/Cosmos3-DROID dataset (LeRobotDataset v3.0) for the
open-loop eval. Rule 8: fetch from upstream, never re-host. Two modes:

  MODE=list      -> list repo files + sizes (no bulk download); prints meta + first data/video files
  MODE=download  -> snapshot_download meta/* + the first NUM_EPISODES data shards & their videos

Env: HF_TOKEN (gated), COSMOS_DROID_REPO (default nvidia/Cosmos3-DROID),
     COSMOS_DROID_DIR (default /models/cosmos3_droid), NUM_EPISODES (default 5).
"""
import os
import sys
from collections import defaultdict

from huggingface_hub import HfApi, snapshot_download


def main() -> int:
    repo = os.environ.get("COSMOS_DROID_REPO", "nvidia/Cosmos3-DROID")
    dest = os.environ.get("COSMOS_DROID_DIR", "/models/cosmos3_droid")
    mode = os.environ.get("MODE", "list")
    n = int(os.environ.get("NUM_EPISODES", "5"))
    token = os.environ.get("HF_TOKEN") or None

    api = HfApi()
    print(f"repo={repo} mode={mode} dest={dest} num_episodes={n}", flush=True)
    info = api.repo_info(repo, repo_type="dataset", files_metadata=True, token=token)
    sibs = info.siblings or []
    # Group by top-level prefix for a quick overview.
    by_prefix = defaultdict(lambda: [0, 0])  # prefix -> [count, bytes]
    for s in sibs:
        pref = s.rfilename.split("/")[0]
        by_prefix[pref][0] += 1
        by_prefix[pref][1] += (s.size or 0)
    print("=== top-level prefixes (count, MB) ===", flush=True)
    for pref, (c, b) in sorted(by_prefix.items()):
        print(f"  {pref:20s} {c:6d} files  {b/1e6:12.1f} MB", flush=True)

    meta = sorted(s.rfilename for s in sibs if s.rfilename.startswith("success/meta/") or s.rfilename.startswith("meta/"))
    data = sorted(s.rfilename for s in sibs if "/data/" in s.rfilename and s.rfilename.endswith(".parquet"))
    videos = sorted(s.rfilename for s in sibs if "/videos/" in s.rfilename and s.rfilename.endswith(".mp4"))
    size_of = {s.rfilename: (s.size or 0) for s in sibs}
    print(f"=== counts: meta={len(meta)} data(parquet)={len(data)} videos(mp4)={len(videos)} ===", flush=True)
    print("--- first meta files ---", flush=True)
    for f in meta[:12]:
        print(f"  {f}  ({size_of[f]/1e6:.2f} MB)", flush=True)
    print("--- first data files ---", flush=True)
    for f in data[:6]:
        print(f"  {f}  ({size_of[f]/1e6:.2f} MB)", flush=True)
    print("--- first video files ---", flush=True)
    for f in videos[:6]:
        print(f"  {f}  ({size_of[f]/1e6:.2f} MB)", flush=True)

    # Build a minimal, split-aware slice: LeRobot v3.0 packs MANY episodes into each ~78MB data
    # parquet and each ~200MB video shard, so one shard per camera already yields dozens of
    # episodes. Fetch success/meta/* (limiting the big episodes parquets to file-000) + the first
    # data shard + the first video shard per camera.
    split = os.environ.get("SPLIT", "success")
    data_shards = int(os.environ.get("NUM_DATA_SHARDS", "1"))
    video_shards = int(os.environ.get("NUM_VIDEO_SHARDS", "1"))

    meta_small = [
        f for f in meta
        if f.startswith(f"{split}/meta/")
        and ("/episodes/" not in f or "/file-000.parquet" in f)
    ]
    data_split = sorted(f for f in data if f.startswith(f"{split}/data/"))[:data_shards]
    cams: dict[str, list[str]] = {}
    for f in videos:
        if f.startswith(f"{split}/videos/"):
            cam = f.split("/videos/")[1].split("/chunk-")[0]
            cams.setdefault(cam, []).append(f)
    video_split: list[str] = []
    for cam, files in sorted(cams.items()):
        video_split += sorted(files)[:video_shards]

    patterns = meta_small + data_split + video_split
    est = sum(size_of.get(p, 0) for p in patterns) / 1e6
    print(f"=== selected split={split}: meta={len(meta_small)} data={len(data_split)} "
          f"videos={len(video_split)} (cams={len(cams)}) ~{est:.1f} MB ===", flush=True)
    for f in patterns:
        print(f"  + {f}  ({size_of.get(f,0)/1e6:.1f} MB)", flush=True)

    if mode != "download":
        print("LIST_DONE (set MODE=download to fetch)", flush=True)
        return 0

    print(f"=== downloading {len(patterns)} files (~{est:.1f} MB) ===", flush=True)
    snapshot_download(
        repo, repo_type="dataset", local_dir=dest,
        allow_patterns=patterns, token=token, max_workers=2,
    )
    print(f"DATASET_FETCH_DONE dest={dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
