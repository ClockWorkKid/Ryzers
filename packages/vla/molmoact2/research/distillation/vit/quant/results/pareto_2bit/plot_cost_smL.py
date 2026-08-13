"""Definitive pre-quantization cost of the S/M/L student vision towers.

Params: exact sum of encoder .numel() from the real distilled fp32 checkpoints.
GMACs : linear/proj = PyTorch dispatch-measured (aten.addmm, exact, incl the final
        projection head); attention = 2*L*T^2*D closed form (SDPA is not dispatch-
        counted). Per single crop of T=729 tokens (14x14 patches), 8 heads, MLP x2.0.
        FLOPs = 2 x GMACs.
"""
from __future__ import annotations
import matplotlib.pyplot as plt

MODELS = ["S", "M", "L"]
XLAB = ["S\n240d x 4blk", "M\n320d x 6blk", "L\n416d x 8blk"]
PARAMS_M = [2.726, 6.098, 12.622]
GMAC_LIN = [1.850, 4.258, 8.951]   # measured (addmm)
GMAC_ATT = [1.020, 2.041, 3.537]   # analytic 2*L*T^2*D
GMAC_TOT = [l + a for l, a in zip(GMAC_LIN, GMAC_ATT)]

CM = ["#4c78a8", "#f58518", "#54a24b"]  # S, M, L
x = range(3)

plt.rcParams.update({"font.size": 11})
fig, (axp, axg) = plt.subplots(1, 2, figsize=(11.4, 5.4), dpi=150)

# --- params ---
bp = axp.bar(x, PARAMS_M, color=CM, width=0.62, edgecolor="black", linewidth=0.6)
for i, v in enumerate(PARAMS_M):
    axp.text(i, v + 0.2, f"{v:.2f}M", ha="center", va="bottom", fontsize=10.5, fontweight="bold")
axp.set_ylabel("Parameters (millions)", fontsize=12)
axp.set_title("Parameter count (fp32, pre-quantization)", fontsize=12)
axp.set_xticks(list(x)); axp.set_xticklabels(XLAB)
axp.set_ylim(0, max(PARAMS_M) * 1.18)
axp.grid(axis="y", alpha=0.3, linestyle="--")
axp.text(0.02, 0.97, f"L/S = {PARAMS_M[2]/PARAMS_M[0]:.1f}x",
         transform=axp.transAxes, va="top", fontsize=9.5, color="#555")

# --- GMACs (stacked: linear measured + attention analytic) ---
axg.bar(x, GMAC_LIN, color=CM, width=0.62, edgecolor="black", linewidth=0.6, label="linear/proj (measured)")
axg.bar(x, GMAC_ATT, bottom=GMAC_LIN, color=CM, width=0.62, alpha=0.45,
        edgecolor="black", linewidth=0.6, hatch="//", label="attention (2LT\u00b2D)")
for i, v in enumerate(GMAC_TOT):
    axg.text(i, v + 0.22, f"{v:.2f}", ha="center", va="bottom", fontsize=10.5, fontweight="bold")
axg.set_ylabel("GMACs  (per 729-token crop)", fontsize=12)
axg.set_title("Compute per forward (fp32, pre-quantization)", fontsize=12)
axg.set_xticks(list(x)); axg.set_xticklabels(XLAB)
axg.set_ylim(0, max(GMAC_TOT) * 1.18)
axg.grid(axis="y", alpha=0.3, linestyle="--")
axg.legend(loc="upper left", fontsize=9, framealpha=0.9)
axg.text(0.98, 0.03, "FLOPs = 2 x GMACs", transform=axg.transAxes,
         ha="right", va="bottom", fontsize=8.5, color="#555")

fig.suptitle("Distilled TinyViT students S / M / L \u2014 model cost before quantization",
             fontsize=13.5, y=1.00)
fig.tight_layout(rect=(0, 0, 1, 0.97))
out = "cost_smL_prequant.png"
fig.savefig(out, bbox_inches="tight")
print("wrote", out)
