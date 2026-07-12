"""Closed-loop LIBERO: causal gate_predict vs teacher gate_distill at matched compute.
Reads artifacts/predict_eval/merged_results.csv, emits a two-panel grouped bar chart.
No pandas dependency (csv module only)."""
import csv, collections, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = os.path.join("artifacts", "predict_eval", "merged_results.csv")
OUT = os.path.join("artifacts", "predict_eval", "predict_vs_distill.png")

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
SLABEL = ["spatial", "object", "goal", "long-10", "AVG"]

# label -> suite -> list of pc
by = collections.defaultdict(lambda: collections.defaultdict(list))
for r in csv.DictReader(open(CSV)):
    if r.get("status") != "done":
        continue
    try:
        by[r["label"]][r["suite"]].append(float(r["pc_success"]))
    except (ValueError, KeyError):
        continue

def series(label):
    vals, allv = [], []
    for s in SUITES:
        v = by[label].get(s, [])
        allv += v
        vals.append(sum(v) / len(v) if v else 0.0)
    vals.append(sum(allv) / len(allv) if allv else 0.0)
    return vals

# (title, distill_label, predict_label)
pairs = [
    ("Pair A  |  FastV L0=3, keep 0.50  (moderate)", "gd050", "pred050"),
    ("Pair B  |  FastV L0=9, keep 0.25  (aggressive)", "gd025", "pred025"),
]

fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
x = range(len(SLABEL))
w = 0.38
for ax, (title, dl, pl) in zip(axes, pairs):
    dv, pv = series(dl), series(pl)
    b1 = ax.bar([i - w/2 for i in x], dv, w, label=f"gate_distill ({dl})", color="#8c9bb0")
    b2 = ax.bar([i + w/2 for i in x], pv, w, label=f"gate_predict ({pl}) — causal, teacher-free", color="#2e7d32")
    for bars in (b1, b2):
        for b in bars:
            ax.annotate(f"{b.get_height():.0f}", (b.get_x() + b.get_width()/2, b.get_height()),
                        ha="center", va="bottom", fontsize=8)
    ax.set_title(title, fontsize=11)
    ax.set_xticks(list(x)); ax.set_xticklabels(SLABEL)
    ax.set_ylim(60, 104)
    ax.axvline(3.5, color="0.8", lw=1, ls="--")
    ax.grid(axis="y", alpha=0.25)
    davg, pavg = dv[-1], pv[-1]
    delta = pavg - davg
    ax.text(0.5, 0.03, f"AVG: distill {davg:.1f}  vs  predict {pavg:.1f}   (Δ {delta:+.1f})",
            transform=ax.transAxes, ha="center", fontsize=10,
            bbox=dict(boxstyle="round", fc="#eef6ee" if delta >= 0 else "#fdeeee", ec="0.7"))
    ax.legend(loc="upper left", fontsize=8)

axes[0].set_ylabel("Closed-loop success rate (%)  —  400 episodes/run")
fig.suptitle("MolmoAct2 seam-6 gate + FastV: causal predictive gating vs teacher distillation (LIBERO, matched compute)",
             fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(OUT, dpi=130)
print("wrote", OUT)
