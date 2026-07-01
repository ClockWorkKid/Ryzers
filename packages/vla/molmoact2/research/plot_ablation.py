import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = os.environ.get("CSV", "/path/to/molmoact2/outputs/ablation_visdrop/results.csv")
OUT = os.environ.get("OUT", "/ryzers/outputs/ablation_visdrop_accuracy.png")

by_suite = defaultdict(list)
with open(CSV) as f:
    for row in csv.DictReader(f):
        by_suite[row["suite"]].append((float(row["keep_frac"]) * 100, float(row["pc_success"])))

order = ["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"]
plt.figure(figsize=(8, 5))
for suite in order:
    pts = sorted(by_suite.get(suite, []))
    if not pts:
        continue
    xs, ys = zip(*pts)
    plt.plot(xs, ys, marker="o", label=suite)

plt.axvline(25, color="gray", ls="--", lw=1, alpha=0.6)
plt.text(25.5, 4, "25% knee", color="gray", fontsize=9)
plt.xlabel("vision tokens kept (%)")
plt.ylabel("closed-loop success (%)")
plt.title("MolmoAct2-Think LIBERO: accuracy vs random vision-token retention")
plt.ylim(0, 103)
plt.xticks([5, 10, 25, 50, 75, 100])
plt.grid(True, alpha=0.3)
plt.legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig(OUT, dpi=120)
print("wrote", OUT)
