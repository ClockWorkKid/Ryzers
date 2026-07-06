"""ROI pre-ViT pruning: analytical compute-saved report for the MolmoAct2 vision
encoder (AMD MI300X training runs).

Numbers are exact for the MolmoAct2-LIBERO ViT config (read from the model
config.json on the cluster):
  input 378x378, patch 14  -> 27x27 = 729 patches/crop, no prefix (CLS) token
  ViT: hidden d=1152, MLP d_ff=4304, heads=16, head_dim=72
  adapter.vit_layers=[-3,-9] -> the backbone runs layers [0..24] = 25 resblocks

Pre-ViT (stage-D) pruning keeps K = round(keep_frac * N) patches through the 25
resblocks, then scatters encoded features back onto the full 729-grid (learned
mask token) so the downstream LLM token count is unchanged. The saving is therefore
in the *vision encoder resblocks*; we report that exactly.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- MolmoAct2-LIBERO ViT config (from config.json) ----
N = 729            # 27x27 patches per crop (378/14)
D = 1152           # ViT hidden size
D_FF = 4304        # ViT MLP intermediate
L = 25             # resblocks actually executed (max(vit_layers)+1 = 24+1)
PATCH_PIX = 14 * 14 * 3  # patch_embedding input dim

KEEP_LEVELS = [1.00, 0.75, 0.50, 0.25]


def resblock_macs(t: int) -> int:
    """MAC count for one ViT transformer layer over t tokens."""
    attn_proj = 4 * t * D * D        # q,k,v,o projections
    attn_score = 2 * t * t * D       # QK^T + attn.V
    mlp = 2 * t * D * D_FF           # up + down (standard 2-layer MLP)
    return attn_proj + attn_score + mlp


def vit_macs(t: int) -> int:
    return L * resblock_macs(t)


def resolve_keep(keep_frac: float) -> int:
    if keep_frac >= 1.0:
        return N
    return max(1, min(N, round(N * keep_frac)))


def main() -> None:
    out_dir = Path(__file__).resolve().parents[3] / "artifacts" / "roi_training"
    out_dir.mkdir(parents=True, exist_ok=True)

    full = vit_macs(N)
    rows = []
    for kf in KEEP_LEVELS:
        k = resolve_keep(kf)
        macs = vit_macs(k)
        red = 1.0 - macs / full
        rows.append(
            {
                "keep_frac": kf,
                "prune_pct": round(100 * (1 - kf)),
                "kept_patches": k,
                "pruned_patches": N - k,
                "vit_resblock_gflops": round(2 * macs / 1e9, 2),  # 2x MAC->FLOP
                "vit_flop_reduction_pct": round(100 * red, 1),
                "vit_speedup_x": round(full / macs, 2),
            }
        )

    report = {
        "model": "MolmoAct2-LIBERO",
        "vision_encoder": {
            "patches_per_crop_N": N,
            "hidden_d": D,
            "mlp_d_ff": D_FF,
            "resblocks_run_L": L,
            "prefix_tokens": 0,
        },
        "note": (
            "Saving is in the ViT resblocks (stage-D pre-ViT pruning). Encoded "
            "features are scattered back onto the full 729-patch grid with a learned "
            "mask token, so downstream LLM token count is unchanged by design."
        ),
        "levels": rows,
    }
    (out_dir / "compute_saved.json").write_text(json.dumps(report, indent=2))

    # ---- bar chart: FLOP reduction % per pruning level ----
    prune = [r["prune_pct"] for r in rows]
    red = [r["vit_flop_reduction_pct"] for r in rows]
    labels = [f"prune {p}%\n(keep {int(100-p)}%)" for p in prune]
    colors = ["#9e9e9e", "#66bb6a", "#42a5f5", "#ef5350"]

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    bars = ax.bar(labels, red, color=colors, edgecolor="black", linewidth=0.6)
    for b, r in zip(bars, rows):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 1.2,
            f"{r['vit_flop_reduction_pct']:.0f}%\n{r['vit_speedup_x']:.2f}x",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )
    ax.set_ylabel("ViT resblock FLOP reduction (%)")
    ax.set_ylim(0, 100)
    ax.set_title(
        "MolmoAct2 ROI pre-ViT pruning — vision-encoder compute saved\n"
        f"N={N} patches/crop, {L} resblocks, d={D}"
    )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "compute_saved.png", dpi=130)
    print("wrote", out_dir / "compute_saved.json")
    print("wrote", out_dir / "compute_saved.png")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
