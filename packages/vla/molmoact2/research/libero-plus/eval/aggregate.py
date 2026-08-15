#!/usr/bin/env python3
"""Aggregate LIBERO-plus eval results into overall + per-suite + per-perturbation + per-difficulty
success rates, using the LIBERO-plus task_classification.json to map each (suite, task_id) to its
perturbation category and difficulty level.

Reads every shard_*/eval_info.json under BASE/<stage>/<kind>/ for each requested stage. Each
eval_info.json holds a "per_task" list; each entry has task_group (suite), task_id, and
metrics.successes (a list of per-episode booleans).

Usage:
  python aggregate.py --base $WORK/libero_plus_eval \
                      --classmap /path/to/LIBERO-plus/libero/libero/benchmark/task_classification.json \
                      --stages s1:rank-16 s2:rank-32 s3:rank-64 s4:rank-128 s5:rank-256 \
                      --kind full            # or: subset
"""
import argparse
import collections
import glob
import json


def load_classmap(path):
    cm = json.load(open(path))

    def lookup(suite, tid):
        lst = cm.get(suite)
        if not lst or tid is None or tid < 0 or tid >= len(lst):
            return (None, None)
        e = lst[tid]
        return (e.get("category"), e.get("difficulty_level"))

    return lookup


def agg(files, lookup):
    n = succ = 0
    suite = collections.defaultdict(lambda: [0, 0])
    cat = collections.defaultdict(lambda: [0, 0])
    diff = collections.defaultdict(lambda: [0, 0])
    for f in files:
        try:
            d = json.load(open(f))
        except Exception:
            continue
        for r in d.get("per_task", []):
            g = r.get("task_group", "?")
            tid = r.get("task_id")
            c, lv = lookup(g, tid)
            for b in r.get("metrics", {}).get("successes", []):
                v = 1 if b else 0
                n += 1; succ += v
                suite[g][0] += 1; suite[g][1] += v
                if c is not None:
                    cat[c][0] += 1; cat[c][1] += v
                if lv is not None:
                    diff[f"L{lv}"][0] += 1; diff[f"L{lv}"][1] += v
    pct = lambda dd: {k: round(100.0 * v[1] / v[0], 1) for k, v in sorted(dd.items()) if v[0]}
    cnt = lambda dd: {k: v[0] for k, v in sorted(dd.items())}
    return {"n": n, "overall": round(100.0 * succ / n, 1) if n else None,
            "per_suite": pct(suite), "per_perturbation": pct(cat),
            "per_difficulty": pct(diff), "counts_perturbation": cnt(cat)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="eval results root (BASE/<stage>/<kind>/shard_*)")
    ap.add_argument("--classmap", required=True, help="LIBERO-plus task_classification.json")
    ap.add_argument("--stages", nargs="+", required=True, help="stage:rank pairs, e.g. s5:rank-256")
    ap.add_argument("--kind", default="full", choices=["full", "subset"])
    args = ap.parse_args()

    lookup = load_classmap(args.classmap)
    out = {}
    for spec in args.stages:
        st, _, rk = spec.partition(":")
        files = glob.glob(f"{args.base}/{st}/{args.kind}/shard_*/eval_info.json")
        out[st] = {"rank": rk or None, **agg(files, lookup)}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
