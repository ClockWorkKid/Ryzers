"""Infer the student architecture actually stored in a distilled checkpoint.

The VARIANTS dims in variants.py must match the trained checkpoint or load will
mismatch. This reads a *_full.pt, extracts the model state, and infers
dim / #conv / #attn / head shape / param count from the tensor shapes.
"""
from __future__ import annotations

import argparse
import torch


def clean(sd: dict) -> dict:
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    out = {}
    for k, v in sd.items():
        nk = k
        for pref in ("module.", "student_model.", "model."):
            if nk.startswith(pref):
                nk = nk[len(pref):]
        out[nk] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    args = ap.parse_args()
    raw = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if isinstance(raw, dict):
        print("[top-level keys]", list(raw.keys())[:10])
    sd = clean(raw)
    # strip the SeamStudent 'encoder.' prefix for readability
    enc = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")} or sd

    tensors = {k: tuple(v.shape) for k, v in enc.items() if hasattr(v, "shape")}
    nparams = sum(v.numel() for v in enc.values() if hasattr(v, "numel"))
    conv = sorted({k.split(".")[1] for k in enc if k.startswith("blocks.") and ".dw." in k}
                  | {k.split(".")[1] for k in enc if k.startswith("blocks.") and (".pw." in k or ".norm." in k)})
    attn = sorted({k.split(".")[1] for k in enc if k.startswith("blocks.") and (".qkv." in k)})
    stem = tensors.get("stem.weight")
    head = {k: v for k, v in tensors.items() if k.startswith("head")}
    print(f"[ckpt] {args.ckpt}")
    print(f"  encoder params = {nparams/1e6:.3f} M")
    print(f"  stem.weight    = {stem}   (dim = {stem[0] if stem else '?'}, in_pixels = {stem[1] if stem else '?'})")
    print(f"  #conv blocks   = {len(conv)}  idx={conv}")
    print(f"  #attn blocks   = {len(attn)} idx={attn}")
    print(f"  head tensors   = {head}")
    # attn head count hint
    for k, v in tensors.items():
        if k.endswith("qkv.weight"):
            print(f"  qkv.weight     = {v}  (3*dim = {v[0]})")
            break
    print(f"  pos            = {tensors.get('pos')}")


if __name__ == "__main__":
    main()
