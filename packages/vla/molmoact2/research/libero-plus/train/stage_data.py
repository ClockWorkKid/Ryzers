#!/usr/bin/env python3
"""Download the LIBERO-plus curriculum-LoRA training assets into a local HuggingFace cache.

Three artifacts, all pulled from public HF repos (rule 8: assets are downloaded by this script,
never vendored in the repo):

  base checkpoint  allenai/MolmoAct2-LIBERO           (model,   ~21 GB)  -> $HF/hub
  training dataset Sylvest/libero_plus_lerobot        (dataset, ~23 GB)  -> $HF/lerobot/hub
  norm assets      lerobot/libero-assets              (dataset, tiny)    -> $HF/hub

The layout mirrors what the containerized lerobot trainer expects with HF_HOME=$HF.

Env:
  HF_ROOT  HF cache root (default: $HOME/molmoact2/hf_cache)
  MAXW     snapshot_download worker count (keep small on shared login nodes; default 4)
  ATTEMPTS resumable retry attempts (default 40)

Usage:
  python stage_data.py            # stage all three
  python stage_data.py MolmoAct2  # stage only repos whose id contains the given substring
"""
import os
import sys
import time

HF = os.environ.get("HF_ROOT", os.path.join(os.path.expanduser("~"), "molmoact2", "hf_cache"))
MAXW = int(os.environ.get("MAXW", "4"))

# Disable the xet backend BEFORE importing huggingface_hub: unauthenticated xet-read-token calls
# get 429-rate-limited on this large (43k-file) dataset. The standard HTTP/LFS path is resumable
# and does not hit the xet token endpoint.
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

from huggingface_hub import snapshot_download

JOBS = [
    ("allenai/MolmoAct2-LIBERO", "model", os.path.join(HF, "hub")),
    ("lerobot/libero-assets", "dataset", os.path.join(HF, "hub")),
    ("Sylvest/libero_plus_lerobot", "dataset", os.path.join(HF, "lerobot", "hub")),
]


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for repo_id, rtype, cache_dir in JOBS:
        if only and only not in repo_id:
            continue
        os.makedirs(cache_dir, exist_ok=True)
        print(f"[stage] {repo_id} ({rtype}) -> {cache_dir}  (max_workers={MAXW}, xet=off)", flush=True)
        attempts = int(os.environ.get("ATTEMPTS", "40"))
        for a in range(1, attempts + 1):
            try:
                p = snapshot_download(repo_id=repo_id, repo_type=rtype,
                                      cache_dir=cache_dir, max_workers=MAXW)
                print(f"[stage] DONE {repo_id} -> {p}", flush=True)
                break
            except Exception as e:  # transient 429/network -> resumable retry with backoff
                wait = min(60, 5 * a)
                print(f"[stage] attempt {a}/{attempts} failed ({type(e).__name__}: {e}); "
                      f"retry in {wait}s", flush=True)
                time.sleep(wait)
        else:
            print(f"[stage] GIVE_UP {repo_id} after {attempts} attempts", flush=True)
            sys.exit(1)
    print("STAGE_ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
