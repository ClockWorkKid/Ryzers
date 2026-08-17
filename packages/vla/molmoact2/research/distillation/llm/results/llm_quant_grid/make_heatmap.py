"""Bit-width heatmap of MolmoAct2 LLM-backbone closed-loop LIBERO success.

Mixed-precision scheme: the ATTENTION MATH (scaled-dot-product attention) is kept
in bf16; the query / key-value / MLP weight projections are fake-quantized to W/A.
Uniform quantization (attention math included) collapses to 0% even at W8A8.

Each cell shows the better of post-training quantization (PTQ) and
quantization-aware training (QAT), with the winning method annotated. QAT wins
only at the 4-bit margin (W4A6, W4A4); PTQ is best everywhere it already works.
FP baseline mean = 99.0.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

W = [8, 6, 4, 3, 2]          # weight bits (rows)
A = [8, 6, 4, 2]             # activation bits (cols)
FP = 99.0
# recommended (best of PTQ/QAT) mean closed-loop success %; None = not run.
# value, method
G = {
    (8, 8): (100.0, "PTQ"),
    (6, 8): (98.8, "PTQ"), (6, 6): (83.8, "PTQ"),
    (4, 8): (95.0, "PTQ"), (4, 6): (75.0, "QAT"), (4, 4): (11.0, "QAT"),
    (3, 8): (0.0, "PTQ"),  (3, 6): (0.0, "PTQ"),  (3, 4): (0.0, "PTQ"),
    (2, 8): (0.0, "PTQ"),  (2, 6): (0.0, "PTQ"),  (2, 4): (0.0, "PTQ"), (2, 2): (0.0, "PTQ"),
}
M = np.full((len(W), len(A)), np.nan)
for i, w in enumerate(W):
    for j, a in enumerate(A):
        if (w, a) in G:
            M[i, j] = G[(w, a)][0]

fig, ax = plt.subplots(figsize=(7, 5.5))
im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
ax.set_xticks(range(len(A)), [f"A{a}" for a in A])
ax.set_yticks(range(len(W)), [f"W{w}" for w in W])
ax.set_xlabel("activation bits")
ax.set_ylabel("weight bits")
ax.set_title("MolmoAct2 LLM-backbone quantization (attention math in bf16)\n"
             f"closed-loop LIBERO success %  |  FP baseline = {FP:.0f}%")
for i in range(len(W)):
    for j in range(len(A)):
        v = M[i, j]
        if np.isnan(v):
            ax.text(j, i, "-", ha="center", va="center", color="gray")
        else:
            meth = G[(W[i], A[j])][1]
            tag = f"\n({meth})" if meth == "QAT" else ""
            ax.text(j, i, f"{v:.0f}{tag}", ha="center", va="center",
                    color="black", fontsize=10, fontweight="bold")
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label("closed-loop success %")
fig.tight_layout()
out = __file__.rsplit("/", 1)[0].rsplit("\\", 1)[0] + "/cl_heatmap.png"
try:
    fig.savefig("cl_heatmap.png", dpi=140)
except Exception:
    pass
fig.savefig(out, dpi=140)
print("wrote", out)
