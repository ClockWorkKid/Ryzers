"""Bit-width heatmap of MolmoAct2 action-expert closed-loop LIBERO success.

Corrected quantization scheme: CROSS-ATTENTION kept high-precision, everything
else (self-attn / MLP / AdaLN modulation / I/O) fake-quantized to W/A. Light
budget: 20 episodes/suite x 4 LIBERO suites. FP baseline mean = 99.0.

W>=3 cells are PTQ-only (no QAT needed); W2 row is QAT-recovered (still the floor
-- flow-matching MSE recovers but closed-loop does not, i.e. 2-bit weights are
genuinely too coarse for the closed-loop velocity field).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

W = [8, 6, 4, 3, 2]          # weight bits (rows)
A = [8, 6, 4, 2]             # activation bits (cols)
FP = 99.0
# mean closed-loop success %; None = not run.  (W>=3: PTQ-only; W2: QAT.)
G = {
    (8, 8): 100.0,
    (6, 8): 100.0, (6, 6): 98.8,
    (4, 8): 96.2,  (4, 6): 96.2, (4, 4): 95.0,
    (3, 8): 96.2,  (3, 6): 96.2, (3, 4): 90.0,
    (2, 8): 23.5,  (2, 6): 23.5, (2, 4): 20.0, (2, 2): 34.5,  # 50 ep/suite x4 (clean)
}
M = np.full((len(W), len(A)), np.nan)
for i, w in enumerate(W):
    for j, a in enumerate(A):
        if (w, a) in G:
            M[i, j] = G[(w, a)]

fig, ax = plt.subplots(figsize=(7, 5.5))
im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
ax.set_xticks(range(len(A)), [f"A{a}" for a in A])
ax.set_yticks(range(len(W)), [f"W{w}" for w in W])
ax.set_xlabel("activation bits")
ax.set_ylabel("weight bits")
ax.set_title(f"MolmoAct2 action-expert quantization (cross-attn FP)\n"
             f"closed-loop LIBERO success %  |  FP baseline = {FP:.0f}%")
for i in range(len(W)):
    for j in range(len(A)):
        v = M[i, j]
        if np.isnan(v):
            ax.text(j, i, "-", ha="center", va="center", color="gray")
        else:
            tag = "" if W[i] >= 3 else "\n(QAT)"
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
