"""Show what the quantized weights actually look like: per-layer integer levels,
per-channel scales, and fp32-vs-quant-grid histograms for a few representative
layers. Writes a small PNG + a JSON summary.

    python -m quant.weight_stats --variant cnn \
        --quant-state resource/ckpt/vit_distill/quant/cnn_w4a6_qat.pt \
        --out-png artifacts/vit_distill/quant/cnn_w4a6_weights.png
"""
from __future__ import annotations

import argparse
import json

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from quant.quant_student import quantize_student_
from quant.variants import load_encoder, resolve


def _int_weight(m):
    """Return (int_tensor, scale) for a Brevitas weight-quant layer, else None."""
    try:
        qw = m.quant_weight()
    except Exception:
        return None
    scale = getattr(qw, "scale", None)
    val = getattr(qw, "value", qw)
    if scale is None:
        return None
    ints = torch.round(val / scale.clamp_min(1e-12)).flatten()
    return ints.detach().float(), scale.detach().float().flatten()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="cnn")
    ap.add_argument("--quant-state", required=True)
    ap.add_argument("--out-png", required=True)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    variant = resolve(args.variant)
    blob = torch.load(args.quant_state, map_location="cpu", weights_only=False)
    wb, ab = blob.get("weight_bits", 4), blob.get("act_bits", 6)

    m = load_encoder(variant, None).to(torch.float32).eval()
    m = quantize_student_(m, weight_bits=wb, act_bits=ab).eval()
    m.load_state_dict(blob["state_dict"], strict=False)

    rows, samples = [], []
    for name, mod in m.named_modules():
        r = _int_weight(mod)
        if r is None:
            continue
        ints, scale = r
        uniq = torch.unique(ints)
        rows.append({
            "layer": name, "type": type(mod).__name__, "n_weights": int(ints.numel()),
            "unique_levels": int(uniq.numel()), "int_min": int(ints.min()),
            "int_max": int(ints.max()), "scale_min": float(scale.min()),
            "scale_max": float(scale.max()), "scale_mean": float(scale.mean()),
        })
        if len(samples) < 4:
            samples.append((name, ints, scale))

    summary = {"variant": variant, "tag": f"W{wb}A{ab}", "mode": blob.get("mode", "?"),
               "n_quant_layers": len(rows),
               "theoretical_max_levels": 2 ** wb,
               "layers": rows}
    print("[weight_stats] " + json.dumps(summary, indent=2)[:2000])
    if args.out_json:
        json.dump(summary, open(args.out_json, "w"), indent=2)

    n = len(samples)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.2))
    if n == 1:
        axes = [axes]
    for ax, (name, ints, scale) in zip(axes, samples):
        lv = int(torch.unique(ints).numel())
        ax.hist(ints.numpy(), bins=min(2 ** wb, 64), color="#3b7dd8", edgecolor="k", lw=0.3)
        ax.set_title(f"{name}\n{lv} levels (max {2**wb})", fontsize=8)
        ax.set_xlabel("integer level"); ax.set_ylabel("count")
    fig.suptitle(f"{variant}  W{wb}A{ab} ({blob.get('mode','?')}) — quantized weight integer levels",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=120)
    print(f"[weight_stats] saved {args.out_png}")


if __name__ == "__main__":
    main()
