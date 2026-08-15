#!/usr/bin/env python3
"""Generate a FULL-coverage LIBERO-plus eval plan (TSV) for one node's GPU workers. Every task_id
of every suite is included (stride=1), sharded into ~SHARD-sized comma-separated id-lists and
spread round-robin across GPUs (gpu = global_shard_idx % GPUS) so each GPU gets a balanced mix of
all suites (keeps the expensive libero_10 long-horizon rollouts from piling on one straggler GPU).

Columns: idx  node  gpu  proc  suite  idlist(comma-separated)
Usage: gen_full_plan.py NODE GPUS SHARD OUT_PATH
  NODE  - placeholder name (the sweep is node-agnostic; it matches on the GPU column only)
  GPUS  - concurrent renders/node (4 is a good osmesa CPU-render contention sweet spot)
  SHARD - target task_ids per shard (~55 => frequent eval_info.json writes for resume-by-shard)
"""
import sys
import math

SUITES = [("libero_spatial", 2402), ("libero_object", 2518),
          ("libero_goal", 2591), ("libero_10", 2519)]


def main():
    node = sys.argv[1]
    gpus = int(sys.argv[2])
    shard = int(sys.argv[3])
    out = sys.argv[4]
    rows = []
    idx = 0
    for suite, n in SUITES:
        ids = list(range(n))
        k = max(1, math.ceil(n / shard))
        for j in range(k):
            chunk = ids[round(j * n / k):round((j + 1) * n / k)]
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
    print(f"wrote {len(rows)} shards -> {out} ; node={node} gpus={gpus} shard={shard} ; full_tasks={tot}")
    for suite, n in SUITES:
        c = sum(len(r[5].split(",")) for r in rows if r[4] == suite)
        s = sum(1 for r in rows if r[4] == suite)
        print(f"  {suite}: {c} of {n} in {s} shards")


if __name__ == "__main__":
    main()
