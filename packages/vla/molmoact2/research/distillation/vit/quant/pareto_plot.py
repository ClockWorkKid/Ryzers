"""Plot the accuracy-vs-precision Pareto knee from calibrate.py output.

Reads one or more ``{variant}_pareto.json`` files (seam cosine / rel-L2 per
bit-width config) and, if present, an optional closed-loop success-rate JSON
(``{variant}_sr.json`` mapping tag -> success rate), then renders a small
knee plot to artifacts/. Kept tiny per the artifact rules.

    python -m quant.pareto_plot --pareto artifacts/vit_distill/quant/cnn_pareto.json \
        --out artifacts/vit_distill/quant/cnn_pareto.png
"""
from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _bits(tag: str) -> int:
    # sort key: total bit budget (weight+act) so the x-axis is monotone-ish
    t = tag.upper()
    return int(t[t.index("W") + 1:t.index("A")]) + int(t[t.index("A") + 1:])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pareto", nargs="+", required=True, help="one or more *_pareto.json")
    ap.add_argument("--sr", nargs="*", default=[], help="optional *_sr.json (tag->success%)")
    ap.add_argument("--out", default="artifacts/vit_distill/quant/pareto.png")
    args = ap.parse_args()

    sr_map: dict[str, dict] = {}
    for s in args.sr:
        sr_map[pathlib.Path(s).stem.replace("_sr", "")] = json.loads(pathlib.Path(s).read_text())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for pj in args.pareto:
        d = json.loads(pathlib.Path(pj).read_text())
        variant = d["variant"]
        rows = sorted(d["sweep"], key=lambda r: _bits(r["tag"]))
        tags = [r["tag"] for r in rows]
        cos = [r["cosine_mean"] for r in rows]
        rel = [r["rel_l2"] for r in rows]
        ax1.plot(tags, cos, marker="o", label=f"{variant} cosine")
        ax2.plot(tags, rel, marker="s", label=f"{variant} rel_L2")
        if variant in sr_map:
            sr = [sr_map[variant].get(t) for t in tags]
            ax1b = ax1.twinx()
            ax1b.plot(tags, sr, marker="^", linestyle="--", color="tab:green", label=f"{variant} SR%")
            ax1b.set_ylabel("closed-loop success %")

    ax1.set_title("Seam cosine vs precision")
    ax1.set_ylabel("cosine (vs fp32 student)")
    ax1.axhline(0.99, color="grey", lw=0.8, ls=":")
    ax1.grid(alpha=0.3); ax1.legend(fontsize=8)
    ax2.set_title("Seam rel-L2 vs precision")
    ax2.set_ylabel("rel_L2"); ax2.grid(alpha=0.3); ax2.legend(fontsize=8)
    for ax in (ax1, ax2):
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    out = pathlib.Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"[pareto] wrote {out}")


if __name__ == "__main__":
    main()
