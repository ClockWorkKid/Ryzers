"""Bounded DROID (lerobot/droid_1.0.1) frame extractor for the hybrid_droid run.

Downloads a small number of exterior-camera MP4 shards, decodes a strided subset of
frames, and writes them as JPEGs to an output dir the distillation reads via
`ImageFolderFrameDataset`. Storage-bounded (a couple shards + ~80k JPEGs ~ 1-2 GB).
Runs online inside the eval SIF on an mi210 node (which has internet + PyAV + PIL).
"""

from __future__ import annotations

import json
import os
import sys

REPO = "lerobot/droid_1.0.1"
OUT = os.environ.get("DROID_OUT", "/cache/droid_frames")
N_SHARDS = int(os.environ.get("DROID_SHARDS", "3"))     # mp4 files (~500MB each)
STRIDE = int(os.environ.get("DROID_STRIDE", "12"))       # keep every Nth frame (~1.25fps @15fps)
MAX_FRAMES = int(os.environ.get("DROID_MAX_FRAMES", "80000"))
JPEG_Q = int(os.environ.get("DROID_JPEG_Q", "90"))


def pick_video_key(info: dict) -> str:
    feats = info.get("features", {})
    vids = [k for k, v in feats.items() if isinstance(v, dict) and v.get("dtype") == "video"]
    if not vids:
        raise SystemExit("no video features found in info.json")
    for pref in ("exterior_1", "exterior", "left", "exterior_2"):
        for k in vids:
            if pref in k:
                return k
    return sorted(vids)[0]


def main() -> None:
    from huggingface_hub import hf_hub_download, snapshot_download

    os.makedirs(OUT, exist_ok=True)
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

    meta_dir = snapshot_download(REPO, repo_type="dataset", allow_patterns=["meta/info.json"])
    info = json.load(open(os.path.join(meta_dir, "meta", "info.json")))
    vkey = pick_video_key(info)
    vtmpl = info["video_path"]  # videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4
    print(f"[droid_prep] repo={REPO} video_key={vkey} shards={N_SHARDS} stride={STRIDE} out={OUT}", flush=True)

    shard_paths = []
    for i in range(N_SHARDS):
        rel = vtmpl.format(video_key=vkey, chunk_index=0, file_index=i)
        try:
            p = hf_hub_download(REPO, rel, repo_type="dataset")
            shard_paths.append(p)
            print(f"[droid_prep] downloaded {rel} -> {p}", flush=True)
        except Exception as e:
            print(f"[droid_prep] skip {rel}: {repr(e)[:120]}", flush=True)

    if not shard_paths:
        raise SystemExit("no DROID shards downloaded")

    import av
    from PIL import Image

    saved = 0
    for sp in shard_paths:
        if saved >= MAX_FRAMES:
            break
        container = av.open(sp)
        stream = container.streams.video[0]
        n = 0
        for frame in container.decode(stream):
            if n % STRIDE == 0:
                img = frame.to_ndarray(format="rgb24")
                Image.fromarray(img).save(
                    os.path.join(OUT, f"droid_{saved:07d}.jpg"), quality=JPEG_Q
                )
                saved += 1
                if saved >= MAX_FRAMES:
                    break
            n += 1
        container.close()
        print(f"[droid_prep] {sp}: total_frames_seen~{n} saved_so_far={saved}", flush=True)

    print(f"[droid_prep] DONE saved={saved} jpgs -> {OUT}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
