"""Render ROI pre-ViT pruning masks on real LIBERO frames.

Two figures:
  mask_viz.png    : per frame -> [original | energy keep50 | content keep50 | energy heatmap]
  mask_levels.png : one frame -> [original | keep75 | keep50 | keep25] (energy selector)

Key finding surfaced: the parameter-free "energy" selector (||patch_embed + pos||)
is dominated by the positional embedding, so it keeps a near-fixed spatial pattern
across frames; the content-only norm (no pos) is mildly image-adaptive. The
learnable gate selector is the content-adaptive path.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ART = Path(__file__).resolve().parents[3] / "artifacts" / "roi_training"
NPZ = ART / "data" / "roi_mask_data.npz"
GRID, P = 27, 14


def overlay(disp, keep_mask):
    """disp: [378,378,3] uint8; keep_mask: [27,27] bool -> dimmed pruned patches."""
    up = np.repeat(np.repeat(keep_mask, P, axis=0), P, axis=1)[:, :, None]  # [378,378,1]
    img = disp.astype(np.float32)
    out = np.where(up, img, img * 0.18 + np.array([10, 10, 60]))
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    d = np.load(NPZ)
    disps = d["disps"]; sel = d["sel"]
    scores = d["scores"]  # full energy
    nf = len(sel)

    # ---- figure 1: energy vs content @ keep50 + heatmap ----
    fig, axes = plt.subplots(nf, 4, figsize=(12, 3 * nf))
    col = ["original frame", "energy keep 50%\n(actual selector)", "content-only keep 50%\n(no pos-emb)", "energy score (per patch)"]
    for r in range(nf):
        axes[r, 0].imshow(disps[r])
        axes[r, 1].imshow(overlay(disps[r], d["full_keep_50"][r]))
        axes[r, 2].imshow(overlay(disps[r], d["content_keep_50"][r]))
        hm = axes[r, 3].imshow(scores[r], cmap="viridis")
        for c in range(4):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(col[c], fontsize=10)
        axes[r, 0].set_ylabel(f"frame {int(sel[r])}", fontsize=9)
    fig.suptitle("MolmoAct2 ROI pre-ViT pruning — masks on real LIBERO frames (keep 50%)\n"
                 "energy selector is positional-dominated (near-fixed pattern); content norm is mildly adaptive",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(ART / "mask_viz.png", dpi=120)
    print("wrote", ART / "mask_viz.png")

    # ---- figure 2: progressive keep levels on one frame ----
    fr = 0
    fig2, ax2 = plt.subplots(1, 4, figsize=(13, 3.6))
    ax2[0].imshow(disps[fr]); ax2[0].set_title("original frame", fontsize=10)
    for i, (kk, lab) in enumerate([("full_keep_75", "keep 75% (prune 25%)"),
                                   ("full_keep_50", "keep 50% (prune 50%)"),
                                   ("full_keep_25", "keep 25% (prune 75%)")]):
        ax2[i + 1].imshow(overlay(disps[fr], d[kk][fr]))
        kept = int(d[kk][fr].sum())
        ax2[i + 1].set_title(f"{lab}\n{kept}/{GRID*GRID} patches -> ViT", fontsize=10)
    for a in ax2:
        a.set_xticks([]); a.set_yticks([])
    fig2.suptitle("ROI pre-ViT pruning — progressive patch keep levels (energy selector)", fontsize=11)
    fig2.tight_layout(rect=(0, 0, 1, 0.92))
    fig2.savefig(ART / "mask_levels.png", dpi=120)
    print("wrote", ART / "mask_levels.png")


if __name__ == "__main__":
    main()
