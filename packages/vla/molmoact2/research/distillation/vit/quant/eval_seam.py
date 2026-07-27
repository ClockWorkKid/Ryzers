"""Measure a quantized student's seam fidelity vs the fp32 student on CLEAN,
held-out LIBERO frames (no augmentation) -- the apples-to-apples metric that
matches the PTQ sweep (calibrate.py). Use it to score QAT-recovered blobs.

    python -m quant.eval_seam --variant cnn \
        --fp32-ckpt resource/ckpt/vit_distill/cnn_fpga_full.pt \
        --quant-state resource/ckpt/vit_distill/quant/cnn_w4a6_qat.pt --n 128
"""
from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

from quant.quant_student import quantize_student_
from quant.variants import load_encoder, resolve

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 8


@torch.no_grad()
def _encode(enc, patches):
    outs = []
    for i in range(0, len(patches), BATCH):
        outs.append(enc(patches[i:i + BATCH].to(DEVICE, torch.float32)).float().cpu())
    return torch.cat(outs, 0)


def load_clean_patches(checkpoint_path: str, n: int) -> torch.Tensor:
    from distill.data import DataConfig, LiberoFrameDataset
    ds = LiberoFrameDataset(
        DataConfig(checkpoint_path=checkpoint_path, patchify_mode="processor"), train=False)
    out = []
    for j in range(min(n, len(ds))):
        p = ds[j][0]
        out.append(p.reshape(-1, p.shape[-2], p.shape[-1]))
    return torch.cat(out, 0).float()[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="cnn")
    ap.add_argument("--fp32-ckpt", required=True)
    ap.add_argument("--quant-state", required=True)
    ap.add_argument("--checkpoint-path", default="allenai/MolmoAct2-LIBERO")
    ap.add_argument("--n", type=int, default=128)
    args = ap.parse_args()

    variant = resolve(args.variant)
    blob = torch.load(args.quant_state, map_location="cpu", weights_only=False)
    wb, ab = blob.get("weight_bits", 4), blob.get("act_bits", 6)

    fp32 = load_encoder(variant, args.fp32_ckpt).to(DEVICE, torch.float32).eval()
    q = load_encoder(variant, None).to(torch.float32).eval()
    q = quantize_student_(q, weight_bits=wb, act_bits=ab).eval()
    miss, unexp = q.load_state_dict(blob["state_dict"], strict=False)
    q = q.to(DEVICE).eval()  # move AFTER load so all scale buffers share device
    print(f"[eval_seam] load miss={len(miss)} unexp={len(unexp)}")

    patches = load_clean_patches(args.checkpoint_path, args.n)
    ref = _encode(fp32, patches)
    out = _encode(q, patches)
    cos = F.cosine_similarity(out.flatten(1), ref.flatten(1), dim=1)
    rel = ((out - ref).norm(dim=-1) / ref.norm(dim=-1).clamp_min(1e-6)).mean()
    res = {"variant": variant, "tag": f"W{wb}A{ab}", "mode": blob.get("mode", "?"),
           "n": int(len(patches)), "clean_cosine_mean": float(cos.mean()),
           "clean_cosine_min": float(cos.min()), "clean_rel_l2": float(rel)}
    print("[eval_seam] " + json.dumps(res))


if __name__ == "__main__":
    main()
