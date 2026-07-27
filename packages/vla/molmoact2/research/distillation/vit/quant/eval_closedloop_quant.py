"""Closed-loop LIBERO launcher with the QUANTIZED student swapped into the ViT.

Same swap mechanism as scratch/closedloop/run_eval_student.py, but the injected
encoder is the Brevitas fake-quant student rebuilt from a quant blob (PTQ or QAT
output of calibrate.py / qat.py). Fake-quant runs in float32 (exact integer
numerics); the seam is cast back to the backbone dtype (bf16) so the frozen
2x2 pool + projector + LLM + action head are untouched.

Requires `brevitas`, the `quant` package and `distill` package on PYTHONPATH in
the eval container (add a thin brevitas/qonnx layer to the lerobot eval image).

Env:
  QUANT_STATE    path to the quant blob (variant/weight_bits/act_bits/state_dict)
  QUANT_VARIANT  cnn|tinyvit|hybrid (default read from blob, else cnn)
All other CLI args forward verbatim to lerobot-eval.
"""
from __future__ import annotations

import importlib
import os
import sys


def _install_quant_patch():
    state = os.environ.get("QUANT_STATE")
    fp32 = os.environ.get("FP32_STUDENT")
    if not state and not fp32:
        print("[quant-swap] QUANT_STATE/FP32_STUDENT unset -> stock teacher ViT", flush=True)
        return
    import torch
    from lerobot.policies.molmoact2.molmoact2_hf_model import modeling_molmoact2 as M
    from quant.quant_student import quantize_student_
    from quant.variants import load_encoder, resolve

    is_fp32 = not state
    if is_fp32:
        # fp32 baseline: same distill.student arch as the quant arms (apples-to-apples),
        # no fake-quant. load_encoder rebuilds the exact variant architecture.
        variant = resolve(os.environ.get("QUANT_VARIANT", "cnn"))
        wbits = abits = None
        blob = None
    else:
        blob = torch.load(state, map_location="cpu", weights_only=False)
        variant = resolve(os.environ.get("QUANT_VARIANT", blob.get("variant", "cnn")))
        wbits, abits = blob.get("weight_bits", 4), blob.get("act_bits", 6)
    cache = {}

    def encode_image(self, images):
        enc = cache.get(id(self))
        if enc is None:
            if is_fp32:
                enc = load_encoder(variant, fp32).to(torch.float32).eval()
                tag = "fp32"
            else:
                enc = load_encoder(variant, None).to(torch.float32).eval()
                enc = quantize_student_(enc, weight_bits=wbits, act_bits=abits).eval()
                miss, unexp = enc.load_state_dict(blob["state_dict"], strict=False)
                if miss or unexp:
                    print(f"[quant-swap] load miss={len(miss)} unexp={len(unexp)}", flush=True)
                tag = f"W{wbits}A{abits}"
            enc = enc.to("cuda" if torch.cuda.is_available() else "cpu").float().eval()
            for p in enc.parameters():
                p.requires_grad_(False)
            cache[id(self)] = enc
            print(f"[quant-swap] {tag} {variant} active on backbone {id(self)}", flush=True)
        # fake-quant runs fp32; cast seam back to the backbone dtype.
        dev = next(enc.parameters()).device
        if images.dim() == 4:
            b, crops, n, p = images.shape
            seam = enc(images.reshape(b * crops, n, p).to(dev, torch.float32))
            seam = seam.reshape(b, crops, n, seam.shape[-1])
        else:
            seam = enc(images.to(dev, torch.float32))
        return seam.to(self.dtype)

    M.MolmoAct2VisionBackbone.encode_image = encode_image
    src = fp32 if is_fp32 else state
    tagp = "fp32" if is_fp32 else f"W{wbits}A{abits}"
    print(f"[quant-swap] patched encode_image <- {src} ({tagp} {variant})", flush=True)


def _run_lerobot_eval():
    from importlib.metadata import entry_points
    try:
        eps = entry_points(group="console_scripts")
    except TypeError:
        eps = entry_points().get("console_scripts", [])
    for ep in eps:
        if ep.name == "lerobot-eval":
            return ep.load()()
    for modname in ("lerobot.scripts.lerobot_eval", "lerobot.scripts.eval"):
        try:
            mod = importlib.import_module(modname)
            fn = getattr(mod, "main", None) or getattr(mod, "eval_main", None)
            if fn:
                return fn()
        except Exception:  # noqa: BLE001
            continue
    raise RuntimeError("could not locate lerobot-eval entry point")


if __name__ == "__main__":
    _install_quant_patch()
    _run_lerobot_eval()
