# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Build stage-D pre-encoder ablation artifacts (curated-style) from a sweep dir.

Reads:
  RAW/results.csv                      (suite,keep_frac,pc_success,...)
  RAW/latency.csv                      (keep_frac,vision_ms,llm_prefill_ms,flow_ms_per_step,total_ms,infer_per_s)
  RAW/run_<suite>_<frac>/eval_info.json (per_task -> task_group/task_id/metrics.successes)

Writes to OUT:
  accuracy_results.csv     normalized per-(suite,frac) success table
  accuracy.png             success vs vision-tokens-kept, per suite
  success_grid.png         per-task success heatmap, one panel per keep_frac
  latency.png              stage-D latency decomposition (vision/prefill/flow) + infer/s
"""
import csv
import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RAW = os.environ.get("RAW", "artifacts/strix/ablation_preenc/raw")
OUT = os.environ.get("OUT", "artifacts/strix/ablation_preenc")
os.makedirs(OUT, exist_ok=True)
ORDER = ["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"]
ACC_TITLE = os.environ.get(
    "ACC_TITLE",
    "MolmoAct2-Think LIBERO: accuracy vs PRE-ENCODER patch retention (stage-D, bf16 fast)",
)
ACC_XLABEL = os.environ.get("ACC_XLABEL", "vision patches kept before encoder (%)")
GRID_TITLE = os.environ.get(
    "GRID_TITLE",
    "Stage-D pre-encoder pruning: per-task closed-loop success (1 ep/task; libero_90 = 15-task subsample)",
)
LAT_TITLE = os.environ.get(
    "LAT_TITLE", "Stage-D: latency decomposition (vision scales, prefill constant)"
)
LAT_XLABEL = os.environ.get("LAT_XLABEL", "vision patches kept before encoder")


def load_results():
    by_suite = defaultdict(dict)  # suite -> {keep: pc}
    p = os.path.join(RAW, "results.csv")
    if not os.path.exists(p):
        return by_suite
    with open(p) as f:
        for row in csv.DictReader(f):
            if not row.get("pc_success"):
                continue
            try:
                by_suite[row["suite"]][float(row["keep_frac"])] = float(row["pc_success"])
            except ValueError:
                continue
    return by_suite


def plot_accuracy(by_suite):
    plt.figure(figsize=(8, 5))
    for suite in ORDER:
        if suite not in by_suite:
            continue
        pts = sorted(by_suite[suite].items())
        xs = [k * 100 for k, _ in pts]
        ys = [v for _, v in pts]
        plt.plot(xs, ys, marker="o", label=suite)
    plt.axvline(25, color="gray", ls="--", lw=1, alpha=0.6)
    plt.text(26, 4, "25% (k=1/4, floor)", color="gray", fontsize=8)
    plt.xlabel(ACC_XLABEL)
    plt.ylabel("closed-loop success (%)")
    plt.title(ACC_TITLE)
    plt.ylim(0, 103)
    plt.xticks([25, 50, 75, 100])
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "accuracy.png"), dpi=120)
    plt.close()


def write_accuracy_csv(by_suite):
    with open(os.path.join(OUT, "accuracy_results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["suite", "keep_frac", "pc_success"])
        for suite in ORDER:
            for k, v in sorted(by_suite.get(suite, {}).items()):
                w.writerow([suite, f"{k:.2f}", v])


def load_per_task():
    # frac -> suite -> {task_id: success(bool)}
    data = defaultdict(lambda: defaultdict(dict))
    for p in glob.glob(os.path.join(RAW, "run_*", "eval_info.json")):
        try:
            info = json.load(open(p))
        except Exception:
            continue
        for t in info.get("per_task", []):
            suite = t.get("task_group")
            tid = t.get("task_id")
            s = t.get("metrics", {}).get("successes") or []
            succ = bool(s[0]) if s else False
            # recover frac from dir name run_<suite>_<frac>
            d = os.path.basename(os.path.dirname(p))
            frac = d.rsplit("_", 1)[-1]
            data[frac][suite][tid] = succ
    return data


def plot_success_grid(per_task):
    fracs = sorted(per_task.keys(), key=float, reverse=True)
    if not fracs:
        return
    suites = [s for s in ORDER if any(s in per_task[f] for f in fracs)]
    max_tasks = max((max(per_task[f][s]) + 1 for f in fracs for s in per_task[f]), default=0)
    cmap = matplotlib.colors.ListedColormap(["#d9534f", "#5cb85c"])
    cmap.set_bad("#e9e9e9")  # not-evaluated tasks (e.g. libero_90 subsample) -> light gray
    fig, axes = plt.subplots(len(fracs), 1, figsize=(min(16, 2 + max_tasks * 0.16), 2.2 * len(fracs)), squeeze=False)
    for r, frac in enumerate(fracs):
        ax = axes[r][0]
        grid = np.full((len(suites), max_tasks), np.nan)
        for i, su in enumerate(suites):
            for tid, succ in per_task[frac].get(su, {}).items():
                grid[i, tid] = 1.0 if succ else 0.0
        ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, vmin=0, vmax=1, aspect="auto")
        ax.set_yticks(range(len(suites)))
        ax.set_yticklabels(suites, fontsize=8)
        ax.set_xticks(range(0, max_tasks, max(1, max_tasks // 20)))
        ax.tick_params(labelsize=7)
        n_ok = int(np.nansum(grid)); n_tot = int(np.sum(~np.isnan(grid)))
        ax.set_title(f"keep={frac}  ({n_ok}/{n_tot} tasks, {100*n_ok/max(n_tot,1):.0f}%)", fontsize=10)
        if r == len(fracs) - 1:
            ax.set_xlabel("task_id (green=success, red=fail, gray=not evaluated)")
    fig.suptitle(GRID_TITLE, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(OUT, "success_grid.png"), dpi=120)
    plt.close(fig)


def plot_latency():
    p = os.path.join(RAW, "latency.csv")
    if not os.path.exists(p):
        return
    rows = list(csv.DictReader(open(p)))
    rows.sort(key=lambda r: float(r["keep_frac"]))
    keep = [float(r["keep_frac"]) * 100 for r in rows]
    vis = [float(r["vision_ms"]) for r in rows]
    pre = [float(r["llm_prefill_ms"]) for r in rows]
    flow = [float(r["flow_ms_per_step"]) * 4 for r in rows]  # eval uses 4 flow steps
    ips = [float(r["infer_per_s"]) for r in rows]
    x = np.arange(len(keep))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, vis, label="vision encode (ViT)", color="#4a90d9")
    ax.bar(x, pre, bottom=vis, label="LLM prefill", color="#f0ad4e")
    ax.bar(x, flow, bottom=np.array(vis) + np.array(pre), label="flow (4 steps)", color="#5cb85c")
    ax.set_xticks(x); ax.set_xticklabels([f"{int(k)}%" for k in keep])
    ax.set_xlabel(LAT_XLABEL)
    ax.set_ylabel("latency (ms)")
    ax.set_title(LAT_TITLE)
    for i in range(len(x)):
        tot = vis[i] + pre[i] + flow[i]
        ax.text(x[i], tot + 6, f"{tot:.0f}ms", ha="center", fontsize=8)
    ax2 = ax.twinx()
    ax2.plot(x, ips, "k--o", label="infer/s")
    ax2.set_ylabel("infer/s")
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "latency.png"), dpi=120)
    plt.close(fig)


def main():
    by_suite = load_results()
    write_accuracy_csv(by_suite)
    plot_accuracy(by_suite)
    plot_success_grid(load_per_task())
    plot_latency()
    print("wrote artifacts to", OUT)


if __name__ == "__main__":
    main()
