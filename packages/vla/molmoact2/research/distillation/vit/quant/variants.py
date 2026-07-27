"""Variant registry + checkpoint loading for the distilled students.

Mirrors the per-variant kwargs in the distillation configs so the quant/export
tools can rebuild the exact architecture a checkpoint was trained with:
  * cnn     -> configs/distill_cnn_fpga.yaml     (FPGA-first target)
  * tinyvit -> configs/distill_siglip_nano.yaml  ("nano siglip")
  * hybrid  -> configs/distill_full.yaml         (the 0.885-cosine baseline)
"""
from __future__ import annotations

import torch

from distill.config import StudentConfig
from distill.student import StudentEncoder, SeamStudent

# variant -> StudentConfig kwargs (kept in sync with the YAML configs).
VARIANTS: dict[str, dict] = {
    "cnn": dict(variant="cnn", dim=384, num_conv_blocks=16, conv_kernel=3),
    "tinyvit": dict(variant="tinyvit", dim=240, num_attn_blocks=4, attn_heads=8, attn_mlp_ratio=2.0),
    "hybrid": dict(variant="hybrid", dim=256, num_conv_blocks=4, num_attn_blocks=3, attn_heads=8),
}

# friendly aliases
ALIASES = {"nano": "tinyvit", "siglip_nano": "tinyvit", "cnn_fpga": "cnn"}


def resolve(variant: str) -> str:
    return ALIASES.get(variant, variant)


def build_encoder(variant: str, **overrides) -> StudentEncoder:
    """Fresh (random-init) StudentEncoder for a named variant."""
    variant = resolve(variant)
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; known={list(VARIANTS)}")
    cfg = StudentConfig(**{**VARIANTS[variant], **overrides})
    return StudentEncoder(cfg)


def load_encoder(variant: str, ckpt_path: str | None, map_location="cpu", **overrides) -> StudentEncoder:
    """Build the encoder and load a distilled checkpoint if provided.

    Checkpoints are torchdistill-saved ``SeamStudent`` state (keys prefixed
    ``encoder.``); we load into a SeamStudent and return its ``.encoder``. If
    ``ckpt_path`` is None the encoder keeps random init (useful for graph/export
    smoke tests that don't need trained weights).
    """
    variant = resolve(variant)
    cfg = StudentConfig(**{**VARIANTS[variant], **overrides})
    wrapper = SeamStudent(cfg)
    if ckpt_path:
        blob = torch.load(ckpt_path, map_location=map_location, weights_only=False)
        sd = blob.get("model", blob.get("state_dict", blob))
        # torchdistill may prefix with 'module.' (DDP); normalize.
        sd = {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in sd.items()}
        missing, unexpected = wrapper.load_state_dict(sd, strict=False)
        if missing:
            print(f"[variants] load {variant}: {len(missing)} missing (first {missing[:3]})")
        if unexpected:
            print(f"[variants] load {variant}: {len(unexpected)} unexpected (first {unexpected[:3]})")
    return wrapper.encoder.to(torch.float32).eval()
