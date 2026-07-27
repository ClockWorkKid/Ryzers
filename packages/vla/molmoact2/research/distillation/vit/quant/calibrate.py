"""PTQ calibration + Pareto precision sweep for the distilled student.

For each bit-width config in the sweep we: (1) deep-copy the fp32 student,
(2) apply Brevitas fake-quant (``quant_student.quantize_student_``),
(3) PTQ-calibrate activation ranges on LIBERO patches
(``brevitas.graph.calibrate.calibration_mode``), (4) measure the seam cosine /
rel-L2 vs the fp32 student on a held-out split, and (5) save the quant state +
metadata.

Calibration source:
  --source dataset   LIBERO frames via distill.data.LiberoFrameDataset (cluster)
  --source file      a pre-captured tensor [N, 729, 588] (patchified pixels)
  --source random    N(0,1) patches (graph/CI smoke only, NOT a real calibration)

Deployment target is W4A6; the sweep {W8A8,W6A6,W4A8,W4A6} maps the knee.

Cluster run (inside the quant image):
    python -m quant.calibrate --variant cnn \
        --ckpt resource/ckpt/vit_distill/cnn_fpga_full.pt \
        --source dataset --sweep W8A8,W6A6,W4A8,W4A6 \
        --out artifacts/vit_distill/quant
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import pathlib

import torch
import torch.nn.functional as F

from brevitas.graph.calibrate import calibration_mode

from quant.quant_student import quantize_student_
from quant.variants import load_encoder, resolve

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CALIB_BATCH = 8


def parse_cfg(tag: str) -> tuple[int, int]:
    """'W4A6' -> (4, 6)."""
    t = tag.upper().replace(" ", "")
    w = int(t[t.index("W") + 1: t.index("A")])
    a = int(t[t.index("A") + 1:])
    return w, a


# --------------------------------------------------------------------------- #
# Calibration patch sources
# --------------------------------------------------------------------------- #
def load_patches(source: str, n: int, checkpoint_path: str, file: str | None) -> torch.Tensor:
    """Return calibration patches shaped [N, 729, 588] (float32)."""
    if source == "random":
        g = torch.Generator().manual_seed(0)
        return torch.randn(n, 729, 588, generator=g)
    if source == "file":
        assert file, "--file required for --source file"
        t = torch.load(file, map_location="cpu", weights_only=False)
        t = t if torch.is_tensor(t) else t["patches"]
        return t.reshape(-1, t.shape[-2], t.shape[-1]).float()[:n]
    if source == "dataset":
        from distill.data import DataConfig, LiberoFrameDataset
        ds = LiberoFrameDataset(
            DataConfig(checkpoint_path=checkpoint_path, patchify_mode="processor"),
            train=False,
        )
        out = []
        for j in range(min(n, len(ds))):
            p = ds[j][0]                       # [crops, N, P]
            out.append(p.reshape(-1, p.shape[-2], p.shape[-1]))
        return torch.cat(out, dim=0).float()[:n]
    raise ValueError(f"bad source {source!r}")


@torch.no_grad()
def _encode(encoder, patches, dtype=torch.float32) -> torch.Tensor:
    outs = []
    for i in range(0, len(patches), CALIB_BATCH):
        b = patches[i:i + CALIB_BATCH].to(DEVICE, dtype)
        outs.append(encoder(b).float().cpu())
    return torch.cat(outs, 0)


def _cosine(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    cos = F.cosine_similarity(a.flatten(1), b.flatten(1), dim=1)
    return float(cos.mean()), float(cos.min())


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(((a - b).norm(dim=-1) / b.norm(dim=-1).clamp_min(1e-6)).mean())


def ptq_one(fp32_encoder, wbits, abits, calib, hold, ref_out) -> dict:
    """Quantize a copy, PTQ-calibrate, and score against the fp32 reference."""
    q = copy.deepcopy(fp32_encoder).to(DEVICE, torch.float32).eval()
    q = quantize_student_(q, weight_bits=wbits, act_bits=abits).to(DEVICE).eval()

    print(f"[ptq] W{wbits}A{abits} calibrating on {len(calib)} patches ...")
    with torch.no_grad(), calibration_mode(q):
        for i in range(0, len(calib), CALIB_BATCH):
            q(calib[i:i + CALIB_BATCH].to(DEVICE, torch.float32))

    q_out = _encode(q, hold)
    cmean, cmin = _cosine(q_out, ref_out)
    rel = _rel_l2(q_out, ref_out)
    print(f"[ptq] W{wbits}A{abits}  cosine mean={cmean:.5f} min={cmin:.5f}  rel_l2={rel:.4f}")
    return {
        "weight_bits": wbits, "act_bits": abits, "tag": f"W{wbits}A{abits}",
        "cosine_mean": cmean, "cosine_min": cmin, "rel_l2": rel,
        "state_dict": q.state_dict(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="PTQ + Pareto precision sweep")
    ap.add_argument("--variant", default="cnn")
    ap.add_argument("--ckpt", default=None, help="distilled checkpoint (omit for smoke)")
    ap.add_argument("--checkpoint-path", default="allenai/MolmoAct2-LIBERO")
    ap.add_argument("--source", default="dataset", choices=["dataset", "file", "random"])
    ap.add_argument("--file", default=None)
    ap.add_argument("--n", type=int, default=512, help="total calib patches")
    ap.add_argument("--hold", type=int, default=64, help="held-out patches for scoring")
    ap.add_argument("--sweep", default="W8A8,W6A6,W4A8,W4A6")
    ap.add_argument("--out", default="artifacts/vit_distill/quant")
    args = ap.parse_args()

    variant = resolve(args.variant)
    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    fp32 = load_encoder(variant, args.ckpt).to(DEVICE, torch.float32).eval()
    patches = load_patches(args.source, args.n + args.hold, args.checkpoint_path, args.file)
    calib, hold = patches[: -args.hold], patches[-args.hold:]
    ref_out = _encode(fp32, hold)   # fp32 student is the reference for quant fidelity
    print(f"[ptq] variant={variant} calib={len(calib)} hold={len(hold)} device={DEVICE}")

    rows = []
    for tag in [t for t in args.sweep.split(",") if t]:
        w, a = parse_cfg(tag)
        res = ptq_one(fp32, w, a, calib, hold, ref_out)
        blob = {k: v for k, v in res.items()}
        blob.update({"variant": variant, "source": args.source, "ckpt": args.ckpt})
        torch.save(blob, outdir / f"{variant}_w{w}a{a}_ptq.pt")
        rows.append({k: res[k] for k in ("tag", "weight_bits", "act_bits",
                                         "cosine_mean", "cosine_min", "rel_l2")})

    summary = {"variant": variant, "source": args.source, "ckpt": args.ckpt, "sweep": rows}
    (outdir / f"{variant}_pareto.json").write_text(json.dumps(summary, indent=2))
    print("[ptq] SUMMARY", json.dumps(rows, indent=2))
    print(f"[ptq] wrote {outdir / (variant + '_pareto.json')}")


if __name__ == "__main__":
    main()
