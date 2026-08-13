"""Plain-QAT (2000-step screen) closed-loop LIBERO success: size x precision.

One curve per quantization level; x-axis = student size (S/M/L). Colors grouped by
weight precision (fp32=black, 4b=blue, 3b=green, 2b=warm), activation shown via
shade/linestyle/marker. Screen = 15 ep/suite x 4 suites (60 ep) -> interim numbers
(pre-qatfix, single seed, under-trained); see CI note in the report.
"""
from __future__ import annotations
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SIZES = ["S", "M", "L"]
XTICKS = ["S\n240d x 4blk", "M\n320d x 6blk", "L\n416d x 8blk"]
X = [0, 1, 2]

# row -> (S, M, L) closed-loop success %  (plain-QAT screen)
DATA = {
    "fp32": [93.3, 96.2, 98.8],
    "W4A6": [96.7, 96.7, 96.7],
    "W4A4": [85.0, 95.0, 95.0],
    "W3A6": [91.7, 98.3, 100.0],
    "W3A4": [75.0, 93.3, 100.0],
    "W2A8": [51.7, 70.0, 85.0],
    "W2A6": [46.7, 73.3, 95.0],
    "W2A4": [16.7, 43.3, 53.3],
    "W2A2": [0.0, 0.0, 0.0],
}

# per-row style: color, linestyle, marker, linewidth
STYLE = {
    "fp32": ("#111111", "-",  "D", 2.6),
    "W4A6": ("#08519c", "-",  "o", 2.0),
    "W4A4": ("#6baed6", "--", "s", 2.0),
    "W3A6": ("#006d2c", "-",  "o", 2.0),
    "W3A4": ("#74c476", "--", "s", 2.0),
    "W2A8": ("#a63603", "-",  "o", 2.0),
    "W2A6": ("#e6550d", "-",  "^", 2.0),
    "W2A4": ("#fd8d3c", "--", "s", 2.0),
    "W2A2": ("#969696", ":",  "x", 1.8),
}

plt.rcParams.update({"font.size": 11, "axes.grid": True,
                     "grid.alpha": 0.3, "grid.linestyle": "--"})
fig, ax = plt.subplots(figsize=(9.2, 6.6), dpi=150)

for name, ys in DATA.items():
    c, ls, mk, lw = STYLE[name]
    ax.plot(X, ys, color=c, linestyle=ls, marker=mk, linewidth=lw,
            markersize=8, label=name, zorder=3,
            markeredgecolor="white", markeredgewidth=0.6)

ax.axhline(97, color="crimson", lw=1.2, ls=(0, (6, 4)), alpha=0.8, zorder=1)
ax.text(2.02, 97.4, "97% target", color="crimson", fontsize=9, va="bottom", ha="right")

ax.set_xticks(X)
ax.set_xticklabels(XTICKS)
ax.set_xlim(-0.15, 2.35)
ax.set_ylim(-4, 104)
ax.set_yticks(range(0, 101, 10))
ax.set_xlabel("Student vision-tower size", fontsize=12, labelpad=8)
ax.set_ylabel("Closed-loop LIBERO success (%)", fontsize=12)
ax.set_title("Plain-QAT screen: closed-loop success vs size & precision\n"
             "TinyViT student  |  4 LIBERO suites x 15 ep (60 ep/config)",
             fontsize=12.5, pad=12)

# main legend (quant levels)
leg1 = ax.legend(title="Quant level (Wt/Act bits)", loc="center left",
                 bbox_to_anchor=(1.01, 0.66), frameon=True, fontsize=9.5,
                 title_fontsize=10)
ax.add_artist(leg1)

# secondary legend: S/M/L definition
sml = [
    Line2D([], [], color="none", label="S = 240 dim x 4 attn-blocks"),
    Line2D([], [], color="none", label="M = 320 dim x 6 attn-blocks"),
    Line2D([], [], color="none", label="L = 416 dim x 8 attn-blocks"),
    Line2D([], [], color="none", label="(8 heads, MLP x2.0; distilled TinyViT)"),
]
ax.legend(handles=sml, title="Size legend", loc="center left",
          bbox_to_anchor=(1.01, 0.20), frameon=True, fontsize=9,
          title_fontsize=10, handlelength=0, handletextpad=0)

fig.tight_layout()
out = "plain_qat_sweep.png"
fig.savefig(out, bbox_inches="tight")
print("wrote", out)
