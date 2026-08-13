"""Size x bit-width closed-loop LIBERO frontier for the distilled TinyViT students.

Each cell = 30 ep x 4 suites = 120 episodes on the 2-epoch qatfix_ep checkpoint.
Annotations show CL% and the Wilson 95% CI half-width (n=120). The 97% target
frontier (>=97.0) is outlined. Sizes carry their fp32 param counts.
"""
from __future__ import annotations
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

SIZES = ["S", "M", "L"]
PARAM = {"S": 2.73, "M": 6.10, "L": 12.62}  # fp32 params (M)
COLS = ["W4A6", "W4A4", "W3A6", "W3A4", "W2A8", "W2A6", "W2A4", "W2A2"]
N = 120  # episodes per cell

# CL% grid (rows S/M/L x cols above), verified from z_epres.sh 24/24
CL = {
    "S": [95.0, 95.8, 95.0, 95.8, 93.3, 93.3, 87.5, 17.5],
    "M": [97.5, 97.5, 95.8, 98.3, 99.2, 97.5, 95.8,  7.5],
    "L": [96.7, 99.2, 99.2, 99.2, 97.5, 98.3, 97.5, 20.8],
}
TARGET = 97.0


def wilson_half(p_pct: float, n: int, z: float = 1.96) -> float:
    p = p_pct / 100.0
    denom = 1 + z * z / n
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return half * 100.0


M = np.array([CL[s] for s in SIZES])
H = np.array([[wilson_half(v, N) for v in CL[s]] for s in SIZES])

fig, ax = plt.subplots(figsize=(13.8, 5.9), dpi=150)
im = ax.imshow(M, cmap="RdYlGn", vmin=60, vmax=100, aspect="auto")

ax.set_xticks(range(len(COLS)))
ax.set_xticklabels(COLS, fontsize=14)
ax.set_yticks(range(len(SIZES)))
ax.set_yticklabels([f"{s}\n{PARAM[s]:.1f}M" for s in SIZES], fontsize=14)
ax.set_xlabel("weight : activation bit-width  (lower = cheaper; 2-bit weights map to LUTs)", fontsize=13)
ax.set_ylabel("student size (fp32 params)", fontsize=13)

for i in range(len(SIZES)):
    for j in range(len(COLS)):
        v = M[i, j]
        txt = "collapse" if v < 60 else f"\u00b1{H[i, j]:.1f}"
        tcol = "white" if v < 70 else "black"
        ax.text(j, i - 0.17, f"{v:.1f}", ha="center", va="center",
                fontsize=17, fontweight="bold", color=tcol)
        ax.text(j, i + 0.27, txt, ha="center", va="center",
                fontsize=12.5, fontweight="bold", color=tcol)
        if v >= TARGET:  # frontier outline
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                    edgecolor="#111", linewidth=2.8))

# lowest-bit config meeting target per size (annotate winners)
for i, s in enumerate(SIZES):
    js = [j for j in range(len(COLS)) if M[i, j] >= TARGET]
    if js:
        j = max(js)  # right-most = lowest precision that still clears 97%
        ax.plot(j, i, marker="*", ms=19, color="gold", mec="black", mew=1.0, zorder=5)

cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
cb.set_label("closed-loop success (%)", fontsize=12)
cb.ax.tick_params(labelsize=11)

ax.set_title("2-bit size\u00d7precision frontier \u2014 closed-loop LIBERO (120 ep/cell, Wilson 95% CI)\n"
             "black outline = \u226597% target;  \u2605 = lowest-bit config meeting target per size",
             fontsize=14)
fig.tight_layout()
out = "frontier_heatmap.png"
fig.savefig(out, bbox_inches="tight")
print("wrote", out)

# also dump the grid as CSV for the report
with open("frontier_grid.csv", "w") as f:
    f.write("size,params_M," + ",".join(COLS) + "\n")
    for s in SIZES:
        f.write(f"{s},{PARAM[s]}," + ",".join(f"{v:.1f}" for v in CL[s]) + "\n")
print("wrote frontier_grid.csv")
