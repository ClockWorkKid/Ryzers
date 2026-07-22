# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""P4 data probe: load ONE real LIBERO episode sample through ImageWAM's own
RobotVideoDataset + ImageWAMProcessor (upstream code, per rule 2.1) and report the
tensor layout/ranges we need to build the open-loop harness. Also validates av1 video
decode on the ROCm image. No model is loaded here.

Uses the same composed cfg as the model (task=libero_flux2_klein_4b_base_imagewam) so the
data settings (num_frames, freq ratio, 2-cam 224x448, norm) match inference. Only
dataset_dirs is overridden to the single suite we have locally, and require_text_cache is
disabled (FLUX.2 encodes text live via Qwen3, so no T5 cache needed).
"""
import os
import sys

import numpy as np
import torch

REPO = os.environ.get("IMAGEWAM_REPO", "/repos/imagewam")
FLUX2_SRC = os.environ.get("FLUX2_SRC", "/repos/flux2")
VARIANT = os.environ.get("FLUX2_VARIANT", "4b")
DATA_DIR = os.environ.get("LIBERO_SUITE_DIR", "/libero_data/libero_object_no_noops_lerobot")
SAMPLE_IDX = int(os.environ.get("SAMPLE_IDX", "0"))

for p in (REPO, os.path.join(FLUX2_SRC, "src"), FLUX2_SRC):
    if p and p not in sys.path:
        sys.path.insert(0, p)


def _compose_cfg():
    from omegaconf import OmegaConf
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    for name, fn in (("eval", eval), ("max", lambda x: max(x)),
                     ("split", lambda s, idx: s.split("/")[int(idx)])):
        try:
            OmegaConf.register_new_resolver(name, fn, replace=True)
        except Exception:
            pass
    overrides = [f"task=libero_flux2_klein_{VARIANT}_base_imagewam"]
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=os.path.join(REPO, "configs"), version_base="1.3"):
        return compose(config_name="sim_libero_omnigen2", overrides=overrides)


def _stats(name, t):
    if isinstance(t, torch.Tensor):
        tf = t.detach().to(torch.float32)
        print(f"  {name:16s} {tuple(t.shape)}  dtype={t.dtype}  "
              f"min={tf.min():.3f} max={tf.max():.3f} mean={tf.mean():.3f}")
    else:
        print(f"  {name:16s} {type(t)}  {t!r}"[:160])


def main() -> int:
    from hydra.utils import instantiate
    from omegaconf import OmegaConf, open_dict

    cfg = _compose_cfg()
    dcfg = cfg.data.train
    print(f"data cfg          : num_frames={int(dcfg.num_frames)}  "
          f"freq_ratio={int(dcfg.action_video_freq_ratio)}  video_size={list(dcfg.video_size)}")

    # FLUX.2 encodes text live via Qwen3, so no precomputed text caches are needed.
    # Use the RELEASED dataset_stats for normalization so proprio/action match how the
    # model was trained (instead of recomputing from just the local suite).
    stats = os.environ.get("DATASET_STATS_PATH",
                           "/models/imagewam_release/libero/flux2_klein_4b/dataset_stats.json")
    with open_dict(dcfg):
        dcfg.dataset_dirs = [DATA_DIR]
        dcfg.require_text_cache = False
        dcfg.qwen_text_cache_dir = None
        dcfg.text_embedding_cache_dir = None
        dcfg.pretrained_norm_stats = stats
    print(f"norm stats        : {stats}")
    ds = instantiate(dcfg, _recursive_=True)
    print(f"dataset           : {type(ds).__name__}  len={len(ds)}  suite={os.path.basename(DATA_DIR)}")

    sample = ds[SAMPLE_IDX]
    print(f"sample[{SAMPLE_IDX}] keys : {sorted(sample.keys())}")
    for k in ("video", "action", "proprio"):
        if k in sample:
            _stats(k, sample[k])
    for k in ("prompt", "instruction"):
        if k in sample:
            _stats(k, sample[k])
    for k in ("image_is_pad", "action_is_pad", "proprio_is_pad"):
        if k in sample and isinstance(sample[k], torch.Tensor):
            print(f"  {k:16s} {tuple(sample[k].shape)}  n_pad={int(sample[k].sum())}")

    # Sanity: finite + expected dims
    v, a, p = sample["video"], sample["action"], sample["proprio"]
    assert torch.isfinite(v).all() and torch.isfinite(a).all() and torch.isfinite(p).all(), "non-finite sample"
    print("PASS: real LIBERO episode decoded + processed OK (av1 decode works on ROCm image)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
