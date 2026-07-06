"""Compare ROI pre-ViT pruning selectors: energy (post-pos-emb norm, position-
dominated) vs content (pre-pos-emb norm, image-adaptive), at matched keep levels.
Reads per-run TSVs pulled from the cluster; one panel per keep fraction.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ART = Path(__file__).resolve().parents[3] / "artifacts" / "roi_training"
DATA = ART / "data"

# (keep label, energy tag, content tag)
LEVELS = [
    ("keep 75% (prune 25%)", "e075", "c075"),
    ("keep 50% (prune 50%)", "e050", "c050"),
    ("keep 25% (prune 75%)", "e025", "c025"),
]
ENERGY_C, CONTENT_C = "#42a5f5", "#ef5350"


def load(tag: str):
    p = DATA / f"{tag}.tsv"
    if not p.exists():
        return np.array([]), np.array([])
    s, y = [], []
    with p.open() as f:
        for row in csv.DictReader(f, delimiter="\t"):
            try:
                s.append(int(row["step"])); y.append(float(row["loss"]))
            except (ValueError, KeyError, TypeError):
                continue
    return np.array(s), np.array(y)


def ema(y, a=0.15):
    if len(y) == 0:
        return y
    o = np.empty_like(y, dtype=float); o[0] = y[0]
    for i in range(1, len(y)):
        o[i] = a * y[i] + (1 - a) * o[i - 1]
    return o


def main():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    rows = []
    for ax, (lab, et, ct) in zip(axes, LEVELS):
        for tag, color, name in ((et, ENERGY_C, "energy"), (ct, CONTENT_C, "content")):
            s, y = load(tag)
            if len(s) == 0:
                continue
            ax.plot(s, y, color=color, alpha=0.18, lw=0.8)
            ax.plot(s, ema(y), color=color, lw=2.0, label=f"{name} ({tag})")
            rows.append(f"{lab} {name}: {len(s)} pts, {y[0]:.2f} -> {y[-1]:.3f} @step{int(s[-1])}")
        ax.set_title(lab, fontsize=10)
        ax.set_xlabel("training step"); ax.grid(alpha=0.3); ax.set_yscale("log")
        ax.legend(fontsize=8, loc="upper right")
    axes[0].set_ylabel("flow-matching loss (log)")
    fig.suptitle("MolmoAct2 ROI pre-ViT pruning — energy (position-dominated) vs content (image-adaptive) selector",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = ART / "energy_vs_content.png"
    fig.savefig(out, dpi=130)
    print("wrote", out)
    print("\n".join(rows))


if __name__ == "__main__":
    main()
