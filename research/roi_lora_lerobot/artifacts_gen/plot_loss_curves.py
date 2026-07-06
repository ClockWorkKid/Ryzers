"""Plot MolmoAct2 ROI pre-ViT pruning training loss curves (convergence per
pruning level). Reads per-run TSVs (step, loss, grdn) pulled from the cluster.
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

RUNS = [
    ("e075", "energy keep 75% (prune 25%)", "#66bb6a"),
    ("e050", "energy keep 50% (prune 50%)", "#42a5f5"),
    ("e025", "energy keep 25% (prune 75%)", "#ef5350"),
    ("g050", "gate  keep 50% (learned)",    "#ab47bc"),
]


def load(tag: str):
    p = DATA / f"{tag}.tsv"
    if not p.exists():
        return np.array([]), np.array([])
    steps, losses = [], []
    with p.open() as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            try:
                steps.append(int(row["step"]))
                losses.append(float(row["loss"]))
            except (ValueError, KeyError, TypeError):
                continue
    return np.array(steps), np.array(losses)


def ema(y: np.ndarray, alpha: float = 0.15) -> np.ndarray:
    if len(y) == 0:
        return y
    out = np.empty_like(y, dtype=float)
    out[0] = y[0]
    for i in range(1, len(y)):
        out[i] = alpha * y[i] + (1 - alpha) * out[i - 1]
    return out


def main() -> None:
    fig, (ax, axl) = plt.subplots(1, 2, figsize=(13, 4.8))
    summary = []
    for tag, label, color in RUNS:
        s, y = load(tag)
        if len(s) == 0:
            continue
        ax.plot(s, y, color=color, alpha=0.22, lw=0.9)
        ax.plot(s, ema(y), color=color, lw=2.0, label=label)
        axl.plot(s, y, color=color, alpha=0.22, lw=0.9)
        axl.plot(s, ema(y), color=color, lw=2.0, label=label)
        summary.append(f"{label}: {len(s)} steps, loss {y[0]:.2f} -> {y[-1]:.3f}")

    for a in (ax, axl):
        a.set_xlabel("training step")
        a.set_ylabel("flow-matching loss")
        a.grid(alpha=0.3)
        a.legend(fontsize=8, loc="upper right")
    ax.set_title("ROI pre-ViT pruning — training loss (linear)")
    axl.set_yscale("log")
    axl.set_title("same, log-y (convergence detail)")
    fig.suptitle("MolmoAct2 LoRA-VLM + ROI pre-ViT pruning on MI300X — loss convergence per pruning level",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = ART / "loss_curves.png"
    fig.savefig(out, dpi=130)
    print("wrote", out)
    print("\n".join(summary))


if __name__ == "__main__":
    main()
