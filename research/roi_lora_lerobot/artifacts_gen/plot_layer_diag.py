"""Per-layer diagnostic for the action-expert->image cross-attention teacher.

For every action-expert layer, plots (a) peakedness (top-10% attention mass) and
(b) content-selectivity = 1 - (fraction of top-8 image tokens shared across all
samples in the probe batch). Sinks show high peakedness but ~0 selectivity (same
tokens every scene); good teacher layers show high selectivity. Motivates the
sink-debiased teacher used for ROI action-attention pruning.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

d = np.load("artifacts/actionattn/teacher_dump.npz")
ii = d["input_ids"]
ip = int(d["image_patch_id"])
AL = d["attn_layers"]  # [B, L, src]
B, L, S = AL.shape

peak = np.zeros(L)
selectivity = np.zeros(L)
for layer in range(L):
    tops = []
    masses = []
    for b in range(B):
        pos = np.where(ii[b] == ip)[0]
        a = AL[b, layer, pos]
        s = np.sort(a)[::-1]
        masses.append(s[: max(1, len(a) // 10)].sum() / (a.sum() + 1e-9))
        tops.append(set(np.argsort(-a)[:8].tolist()))
    peak[layer] = np.mean(masses)
    shared = len(set.intersection(*tops))
    selectivity[layer] = 1.0 - shared / 8.0  # 8 = top-k

fig, ax = plt.subplots(figsize=(13, 4.5))
x = np.arange(L)
ax.bar(x - 0.2, peak, width=0.4, label="peakedness (top-10% mass)", color="#4C72B0")
ax.bar(x + 0.2, selectivity, width=0.4, label="content-selectivity (1 - shared top-8)", color="#DD8452")
ax.axhline(0.1, ls=":", c="gray", lw=1)
ax.set_xticks(x)
ax.set_xticklabels(x, fontsize=7)
ax.set_xlabel("action-expert layer")
ax.set_ylim(0, 1.05)
ax.set_title("Action-expert cross-attention per layer: sinks (peaked but 0-selective, e.g. L34) vs "
             "content layers (peaked AND selective, e.g. L9/20/21)")
ax.legend(loc="upper left")
for layer in (9, 20, 21):
    ax.annotate("teacher", (layer, selectivity[layer]), textcoords="offset points", xytext=(0, 4),
                ha="center", fontsize=7, color="#DD8452")
ax.annotate("sink", (34, peak[34]), textcoords="offset points", xytext=(0, 4), ha="center",
            fontsize=7, color="#4C72B0")
fig.tight_layout()
out = Path("artifacts/actionattn/layer_diagnostic.png")
fig.savefig(out, dpi=130)
print(f"wrote {out}")
