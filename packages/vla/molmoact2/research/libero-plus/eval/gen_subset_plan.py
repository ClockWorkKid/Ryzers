#!/usr/bin/env python3
"""Generate a STRATIFIED-SUBSET LIBERO-plus eval plan (TSV). Per suite we take ~TARGET*suite_frac
task_ids by UNIFORM STRIDE over the suite's id range. Because each suite's perturbation dimensions
occupy contiguous id sub-ranges, uniform-stride sampling is proportional across all 7 perturbation
dimensions + 5 difficulty levels -> a representative subset whose per-suite / per-dim success rates
estimate the full benchmark (validated: full-vs-subset gap < 1 pt on the r128/r256 stages).

Columns: idx  node  gpu  proc  suite  idlist(comma-separated)
Usage: gen_subset_plan.py NODE TARGET GPUS PPG OUT_PATH
  TARGET - total subset size across suites (e.g. 1673)
  GPUS   - concurrent renders/node ; PPG - shards per GPU run sequentially
"""
import sys

SUITES = [("libero_spatial", 2402), ("libero_object", 2518),
          ("libero_goal", 2591), ("libero_10", 2519)]


def main():
    node = sys.argv[1]; target = int(sys.argv[2]); gpus = int(sys.argv[3])
    ppg = int(sys.argv[4]); out = sys.argv[5]
    total = sum(n for _, n in SUITES)
    workers = gpus * ppg
    suite_ids = {}
    for suite, n in SUITES:
        k = max(1, round(target * n / total))
        stride = max(1, round(n / k))
        suite_ids[suite] = list(range(0, n, stride))[:k]
    nS = len(SUITES)
    base = workers // nS; rem = workers - base * nS
    sps = [base + (1 if i < rem else 0) for i in range(nS)]
    rows = []; idx = 0
    for (suite, n), k in zip(SUITES, sps):
        ids = suite_ids[suite]
        k = max(1, min(k, len(ids)))
        for j in range(k):
            chunk = ids[round(j * len(ids) / k):round((j + 1) * len(ids) / k)]
            if not chunk:
                continue
            gpu = idx % gpus
            proc = idx // gpus
            rows.append((idx, node, gpu, proc, suite, ",".join(map(str, chunk))))
            idx += 1
    with open(out, "w") as f:
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")
    tot = sum(len(r[5].split(",")) for r in rows)
    print(f"wrote {len(rows)} shards -> {out} ; node={node} target={target} gpus={gpus} ppg={ppg} ; subset_tasks={tot}")
    for suite, n in SUITES:
        c = sum(len(r[5].split(",")) for r in rows if r[4] == suite)
        print(f"  {suite}: {c} of {n}")


if __name__ == "__main__":
    main()
