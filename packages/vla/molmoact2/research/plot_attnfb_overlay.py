# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Two-column attention-feedback saliency overlay (rule 2b: reference image left,
prediction/saliency right). Reads saliency.npy + frame_ext.png dumped by
validate_attnfeedback.py and renders the per-pooling-group saliency for the
exterior crop on top of the input frame, so it can be eyeballed against the
trusted gradient/occlusion overlays already in artifacts/strix/interp/.

    python plot_attnfb_overlay.py [in_dir] [out_png]
"""
import os
import sys

import numpy as np

IN = sys.argv[1] if len(sys.argv) > 1 else "/ryzers/outputs/attnfb/validate"
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(IN, "attnfb_overlay.png")


def grid_for(n_per_crop):
    s = int(round(n_per_crop ** 0.5))
    return s if s * s == n_per_crop else None


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    sal = np.load(os.path.join(IN, "saliency.npy")).astype(np.float64).reshape(-1)
    img = Image.open(os.path.join(IN, "frame_ext.png")).convert("RGB")
    W, H = img.size

    # crop 0 (exterior) occupies the first 1/ncrop of the tokens; infer a square grid
    n = sal.size
    crop0 = sal
    for ncrop in (3, 2, 1):
        if n % ncrop == 0 and grid_for(n // ncrop):
            g = grid_for(n // ncrop)
            crop0 = sal[: n // ncrop]
            break
    else:
        g = grid_for(n) or int(np.floor(np.sqrt(n)))
        crop0 = sal[: g * g]
    grid = crop0.reshape(g, g)
    grid = (grid - grid.min()) / (grid.ptp() + 1e-9)

    heat = np.asarray(Image.fromarray((grid * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)) / 255.0

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img); ax[0].set_title("input (exterior)"); ax[0].axis("off")
    ax[1].imshow(img)
    ax[1].imshow(heat, cmap="jet", alpha=0.5)
    ax[1].set_title("attn-feedback saliency (debiased, per 2x2 group)")
    ax[1].axis("off")
    fig.suptitle(f"MolmoAct2 attention-feedback pruning saliency  (grid {g}x{g}, n={n})")
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=110, bbox_inches="tight")
    print("wrote", OUT)


if __name__ == "__main__":
    raise SystemExit(main())
