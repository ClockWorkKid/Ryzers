"""Visualize the action-expert -> image cross-attention (the ROI "teacher" signal).

Loads the teacher_dump.npz produced by a full (unpruned) forward with
ROI_DUMP_TEACHER=1 and maps the per-image-token attention back onto the 729-patch
ViT grid for each image, overlaying it on the reconstructed frame. This answers
the Phase-0 question: does the action expert focus on task-relevant regions
(gripper / target object) rather than the fixed positional hotspots that the
energy selector keys on?
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PATCHES = 729           # 27x27 SigLIP2 patch grid per image
GRID = 27
PATCH_PX = 14           # 14x14 px per patch
POOL = 4                # 2x2 pooling


def reconstruct_image(pv_img: np.ndarray) -> np.ndarray:
    """pv_img: [729, 588] -> [378, 378, 3] display image (min-max normalized)."""
    p = pv_img.reshape(PATCHES, 3, PATCH_PX, PATCH_PX).transpose(0, 2, 3, 1)  # [729,14,14,3]
    canvas = np.zeros((GRID * PATCH_PX, GRID * PATCH_PX, 3), dtype=np.float32)
    for idx in range(PATCHES):
        r, c = divmod(idx, GRID)
        canvas[r * PATCH_PX:(r + 1) * PATCH_PX, c * PATCH_PX:(c + 1) * PATCH_PX] = p[idx]
    lo, hi = canvas.min(), canvas.max()
    return (canvas - lo) / (hi - lo + 1e-6)


def patch_attn_for_sample(attn_row, img_positions, pooling_rows, n_images):
    """Spread per-pooled-token attention onto per-patch scores, split by image.

    The two camera images reuse the same 0..728 patch-index range, so images are
    separated by pooled-token *order* (the sequence lists image 0's pooled tokens
    then image 1's), not by index value. Returns dict img_idx -> [729] patch attn.
    """
    tok_attn = attn_row[img_positions]                 # [n_tok]
    n_tok = len(tok_attn)
    per_img_tok = n_tok // max(1, n_images)
    per_img: dict[int, np.ndarray] = {}
    counts: dict[int, np.ndarray] = {}
    for k, a in enumerate(tok_attn):
        img = min(k // per_img_tok, n_images - 1)
        for p in pooling_rows[k]:
            if p < 0:
                continue
            local = int(p) % PATCHES
            per_img.setdefault(img, np.zeros(PATCHES, np.float64))
            counts.setdefault(img, np.zeros(PATCHES, np.float64))
            per_img[img][local] += float(a)
            counts[img][local] += 1.0
    for img in per_img:
        per_img[img] /= np.maximum(counts[img], 1.0)
    return per_img


def upsample(grid_vals: np.ndarray) -> np.ndarray:
    g = grid_vals.reshape(GRID, GRID)
    return np.kron(g, np.ones((PATCH_PX, PATCH_PX)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="artifacts/actionattn/teacher_dump.npz")
    ap.add_argument("--out", default="artifacts/actionattn/teacher_attention.png")
    ap.add_argument("--keep_frac", type=float, default=0.25)
    ap.add_argument("--layers", default="34", help="comma list of action-expert layers to average, or 'mean' for all")
    args = ap.parse_args()

    d = np.load(args.dump)
    input_ids = d["input_ids"]             # [B, src]
    img_id = int(d["image_patch_id"])
    pooling = d["image_token_pooling"]     # [B*n_tok, POOL]
    pv = d["pixel_values"]                 # [n_dumped_images, 729, 588]
    n_images = int(pv.shape[0])

    AL = d["attn_layers"]                          # [B, n_layers, src]
    if args.layers == "mean":
        attn = AL.mean(axis=1)
        layer_desc = "mean of all layers"
    elif args.layers.startswith("debias"):
        # remove the fixed positional/sink template (cross-sample mean per layer),
        # keep the content-specific positive residual, sum over selected layers.
        spec = args.layers.split(":", 1)
        sel = [int(x) for x in spec[1].split(",")] if len(spec) > 1 else list(range(AL.shape[1]))
        resid = np.clip(AL - AL.mean(axis=0, keepdims=True), 0, None)
        attn = resid[:, sel, :].sum(axis=1)
        layer_desc = f"sink-debiased, layers {sel}"
    else:
        sel = [int(x) for x in args.layers.split(",")]
        attn = AL[:, sel, :].mean(axis=1)
        layer_desc = f"layer(s) {sel}"

    b = 0
    img_positions = np.where(input_ids[b] == img_id)[0]
    n_tok = len(img_positions)
    pooling_b = pooling[b * n_tok:(b + 1) * n_tok]     # sample-0 pooled tokens
    per_img = patch_attn_for_sample(attn[b], img_positions, pooling_b, n_images)

    # global attention stats over image tokens (peakedness)
    a = attn[b][img_positions]
    a_norm = a / (a.sum() + 1e-9)
    ent = float(-(a_norm * np.log(a_norm + 1e-12)).sum())
    max_ent = float(np.log(n_tok))
    order = np.argsort(-a)
    top10 = float(a[order[: max(1, n_tok // 10)]].sum() / (a.sum() + 1e-9))
    print(f"[stats] img_tokens={n_tok} entropy={ent:.2f}/{max_ent:.2f} "
          f"({ent/max_ent:.2%} of uniform)  top-10% mass={top10:.2%}")

    img_ids = sorted(per_img.keys())
    n_imgs = min(len(img_ids), pv.shape[0])
    names = {0: "main camera", 1: "wrist camera"}
    fig, axes = plt.subplots(n_imgs, 3, figsize=(11, 4.2 * n_imgs), squeeze=False)
    for row, im in enumerate(img_ids[:n_imgs]):
        frame = reconstruct_image(pv[im])
        pa = per_img[im]
        heat = upsample(pa / (pa.max() + 1e-9))
        # top-keep mask on this image's patches
        k = max(1, int(round(args.keep_frac * PATCHES)))
        thr = np.sort(pa)[::-1][k - 1]
        keep = (pa >= thr).astype(np.float32)
        keep_up = upsample(keep)

        axes[row][0].imshow(frame)
        axes[row][0].set_title(f"{names.get(im, f'image {im}')} (frame)")
        axes[row][1].imshow(frame)
        axes[row][1].imshow(heat, cmap="jet", alpha=0.55)
        axes[row][1].set_title("action-expert attention (overlay)")
        axes[row][2].imshow(frame)
        axes[row][2].imshow(keep_up, cmap="Greens", alpha=0.45)
        axes[row][2].set_title(f"kept patches @ keep={args.keep_frac:.0%}")
        for c in range(3):
            axes[row][c].axis("off")

    fig.suptitle(
        f"Action-expert cross-attention over image tokens ({layer_desc})  |  "
        f"entropy {ent/max_ent:.0%} of uniform, top-10% tokens = {top10:.0%} of mass",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
