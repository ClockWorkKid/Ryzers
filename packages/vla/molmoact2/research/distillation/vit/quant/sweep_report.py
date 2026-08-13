"""Aggregate the 2-bit size-vs-precision sweep into a heatmap + tables.

Reads a long-form CSV of closed-loop results (one row per size x bit-width, plus
optional fp32 anchors and FINN resource estimates) and emits:
  * HEATMAP.png  -- (size rows x W/A columns) CL success grid, >=97 frontier drawn
  * a markdown results table (printed / written to REPORT_TABLE.md)

Kept dependency-light (pandas optional): pure csv + matplotlib. Runs on the
laptop; artifacts stay small (rule 4).

CSV schema (header required):
    size,weight_bits,act_bits,cl_mean,seam_cos,stage,lut,dsp,notes
`size` in {S,M,L}; cl_mean in [0,100] (blank if not yet run). fp32 anchors use
weight_bits=32,act_bits=32.

    python -m quant.sweep_report --csv artifacts/quant_sweep_2bit/results.csv \
        --out artifacts/quant_sweep_2bit
"""
from __future__ import annotations

import argparse
import csv
import pathlib

SIZE_ORDER = ["S", "M", "L"]
# W/A columns in frontier order (coarse -> fine precision left->right).
WA_ORDER = ["W2A2", "W2A4", "W2A6", "W2A8", "W3A4", "W3A6", "W4A4", "W4A6"]


def load(csv_path: pathlib.Path) -> list[dict]:
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def _tag(r: dict) -> str:
    return f"W{r['weight_bits']}A{r['act_bits']}"


def build_grid(rows: list[dict]) -> dict[tuple[str, str], float]:
    grid: dict[tuple[str, str], float] = {}
    for r in rows:
        if r.get("weight_bits") in ("32", "") or not r.get("cl_mean"):
            continue
        try:
            v = float(r["cl_mean"])
        except ValueError:
            continue
        key = (r["size"], _tag(r))
        grid[key] = max(grid.get(key, float("-inf")), v)   # keep best per cell
    return grid


def markdown_table(rows: list[dict]) -> str:
    grid = build_grid(rows)
    hdr = "| size | " + " | ".join(WA_ORDER) + " |"
    sep = "|" + "---|" * (len(WA_ORDER) + 1)
    lines = [hdr, sep]
    for s in SIZE_ORDER:
        cells = []
        for wa in WA_ORDER:
            v = grid.get((s, wa))
            cells.append(f"{v:.1f}" if v is not None else "-")
        lines.append(f"| {s} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def make_heatmap(rows: list[dict], out: pathlib.Path, thresh: float = 97.0) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:  # noqa: BLE001
        print(f"[sweep_report] matplotlib unavailable ({e!r}); skipping heatmap")
        return
    grid = build_grid(rows)
    mat = np.full((len(SIZE_ORDER), len(WA_ORDER)), np.nan)
    for i, s in enumerate(SIZE_ORDER):
        for j, wa in enumerate(WA_ORDER):
            if (s, wa) in grid:
                mat[i, j] = grid[(s, wa)]
    fig, ax = plt.subplots(figsize=(9, 3.2))
    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=40, vmax=100)
    ax.set_xticks(range(len(WA_ORDER)), WA_ORDER, rotation=45, ha="right")
    ax.set_yticks(range(len(SIZE_ORDER)), [f"{s}" for s in SIZE_ORDER])
    ax.set_xlabel("weight/activation bits (coarse -> fine)")
    ax.set_ylabel("model size")
    ax.set_title(f"TinyViT 2-bit sweep: closed-loop success (>= {thresh:.0f} = frontier)")
    for i in range(len(SIZE_ORDER)):
        for j in range(len(WA_ORDER)):
            if not np.isnan(mat[i, j]):
                v = mat[i, j]
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        color="white" if v < 80 else "black", fontsize=8)
                if v >= thresh:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                               edgecolor="red", lw=2))
    fig.colorbar(im, ax=ax, label="CL success (%)")
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "HEATMAP.png", dpi=130)
    print(f"[sweep_report] wrote {out / 'HEATMAP.png'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="artifacts/quant_sweep_2bit")
    ap.add_argument("--thresh", type=float, default=97.0)
    args = ap.parse_args()
    rows = load(pathlib.Path(args.csv))
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tbl = markdown_table(rows)
    (out / "REPORT_TABLE.md").write_text("# 2-bit sweep CL success (%)\n\n" + tbl + "\n")
    print(tbl)
    make_heatmap(rows, out, thresh=args.thresh)


if __name__ == "__main__":
    main()
