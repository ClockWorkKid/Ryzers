"""DAgger mixed batch source: interleave teacher-relabeled on-policy samples with demo batches.

The relabeled samples (from ``relabel_teacher.py``) are B=1 model-ready dicts carrying the student-
visited model inputs + the teacher's current-step ACTION + pad masks -- byte-compatible with what
``policy.forward`` consumes. We collate ``args.batch`` of them into a rectangular batch (the shared
pre-processor pads all samples to a fixed length, so demo and on-policy batches have identical
shapes) and yield either an on-policy batch (prob ``mix``) or a demo batch (prob ``1-mix``).
"""

from __future__ import annotations
import glob
import os
import random
import torch


def _as_dir_list(relabel_dir):
    """Accept a single dir, a comma/os.pathsep-separated string, or a list of dirs.
    Enables DAgger data AGGREGATION across rounds (true DAgger D <- D u new)."""
    if isinstance(relabel_dir, (list, tuple)):
        dirs = list(relabel_dir)
    else:
        parts = str(relabel_dir).replace(os.pathsep, ",").split(",")
        dirs = [p.strip() for p in parts if p.strip()]
    return dirs


def load_relabel_samples(relabel_dir):
    """Load relabeled on-policy samples from one or more dirs (aggregation-friendly)."""
    samples = []
    for d in _as_dir_list(relabel_dir):
        files = sorted(glob.glob(os.path.join(d, "*.pt")))
        for f in files:
            shard = torch.load(f, map_location="cpu", weights_only=False)
            samples.extend(shard.get("samples", []))
    return samples


def _sig(sample):
    """Shape signature of all tensor fields -- on-policy states from different tasks have different
    sequence lengths, so we can only cat samples that share every tensor shape."""
    return tuple((k, tuple(v.shape)) for k, v in sample.items() if torch.is_tensor(v))


def _bucket_by_sig(pool):
    buckets = {}
    for s in pool:
        buckets.setdefault(_sig(s), []).append(s)
    return buckets


def _collate(samples, device):
    out = {}
    for k in samples[0]:
        vals = [s[k] for s in samples]
        if torch.is_tensor(vals[0]):
            out[k] = torch.cat(vals, 0).to(device)          # each sample is [1, ...]
        else:
            merged = []
            for v in vals:
                merged.extend(v if isinstance(v, list) else [v])
            out[k] = merged
    return out


def iter_dagger_mixed(args, device, relabel_dir, mix=0.5, seed=0):
    """Infinite iterator of mixed batches. ``mix`` = fraction of on-policy (relabeled) batches.

    On-policy samples are bucketed by tensor-shape signature (sequence length varies per task); a
    batch is drawn from a single bucket so every sample is cat-compatible -- no padding, no change
    to model behavior. Demo batches come from the fixed-length dataset loader as before.
    """
    import data as D
    demo_iter = D.iter_full_batches(args, device)
    dirs = _as_dir_list(relabel_dir)
    pool = load_relabel_samples(dirs)
    if not pool:
        raise RuntimeError(f"[dagger_data] no relabel samples found in {dirs}")
    buckets = _bucket_by_sig(pool)
    keys = list(buckets.keys())
    weights = [len(buckets[k]) for k in keys]
    sizes = sorted(weights, reverse=True)
    print(f"[dagger_data] on-policy pool={len(pool)} states from {len(dirs)} dir(s) "
          f"in {len(keys)} shape-buckets (largest={sizes[:5]}) | mix={mix} | batch={args.batch}", flush=True)
    rng = random.Random(seed)
    B = int(args.batch)
    while True:
        if rng.random() < mix:
            bk = rng.choices(keys, weights=weights, k=1)[0]
            grp = buckets[bk]
            picks = [grp[rng.randrange(len(grp))] for _ in range(B)]
            yield _collate(picks, device)
        else:
            yield next(demo_iter)
