"""Analytic FLOP profiler for the teacher ViT and the student encoder.

Pure Python (stdlib only) so it runs on the laptop without torch. Counts are in
MACs; GFLOPs are reported as 2 * MACs / 1e9 (one multiply-add = 2 FLOPs), the
same convention as ``research/roi_lora_lerobot/artifacts_gen/compute_saved.py``
(teacher full ViT ~= 616 GFLOPs/crop, which this reproduces exactly).

Usage (local):
    python -m distill.flops                     # default hybrid student vs teacher
    python -m distill.flops --variant cnn --dim 320 --num-conv-blocks 6
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from .config import StudentConfig, TeacherSpec, TEACHER_LIBERO


# --------------------------------------------------------------------------- #
# Primitive MAC counters
# --------------------------------------------------------------------------- #
def linear_macs(tokens: int, in_dim: int, out_dim: int) -> int:
    """Dense linear / 1x1 conv over ``tokens`` positions."""
    return tokens * in_dim * out_dim


def depthwise_conv_macs(tokens: int, channels: int, kernel: int) -> int:
    """Depthwise KxK conv over ``tokens`` positions (padding='same')."""
    return tokens * channels * kernel * kernel


def attn_block_macs(tokens: int, dim: int, mlp_ratio: float) -> int:
    """One pre-norm transformer block: q/k/v/o projections + scores + MLP."""
    qkvo = 4 * tokens * dim * dim
    scores = 2 * tokens * tokens * dim          # QK^T and attn @ V
    mlp = 2 * tokens * dim * int(round(mlp_ratio * dim))
    return qkvo + scores + mlp


# --------------------------------------------------------------------------- #
# Teacher
# --------------------------------------------------------------------------- #
def teacher_macs(spec: TeacherSpec = TEACHER_LIBERO) -> dict:
    n = spec.num_patches
    patch_embed = linear_macs(n, spec.patch_pixels, spec.hidden_size)
    per_block = attn_block_macs(n, spec.hidden_size, spec.mlp_hidden / spec.hidden_size)
    blocks = spec.resblocks_run * per_block
    total = patch_embed + blocks
    return {
        "patch_embed": patch_embed,
        "resblocks": blocks,
        "total": total,
        "gflops": round(2 * total / 1e9, 2),
    }


# --------------------------------------------------------------------------- #
# Student
# --------------------------------------------------------------------------- #
def student_macs(cfg: StudentConfig) -> dict:
    n = cfg.num_tokens
    parts: dict[str, int] = {}

    # Stem: project raw per-patch pixels -> working width (1x1 over the grid).
    parts["stem"] = linear_macs(n, cfg.in_pixels, cfg.dim)

    if cfg.variant in {"hybrid", "cnn"}:
        # Each conv block: depthwise KxK + pointwise 1x1 (MobileNet-style) + a
        # second pointwise to mix (invert-bottleneck kept simple at ratio 1).
        per_conv = (
            depthwise_conv_macs(n, cfg.dim, cfg.conv_kernel)
            + linear_macs(n, cfg.dim, cfg.dim)
        )
        parts["conv_blocks"] = cfg.num_conv_blocks * per_conv

    if cfg.variant in {"hybrid", "tinyvit"}:
        per_attn = attn_block_macs(n, cfg.dim, cfg.attn_mlp_ratio)
        parts["attn_blocks"] = cfg.num_attn_blocks * per_attn

    # Head: working width -> seam_dim (optionally 2-layer).
    if cfg.head_hidden and cfg.head_hidden > 0:
        parts["head"] = linear_macs(n, cfg.dim, cfg.head_hidden) + linear_macs(
            n, cfg.head_hidden, cfg.seam_dim
        )
    else:
        parts["head"] = linear_macs(n, cfg.dim, cfg.seam_dim)

    total = sum(parts.values())
    parts["total"] = total
    parts["gflops"] = round(2 * total / 1e9, 3)
    return parts


def profile(cfg: StudentConfig, spec: TeacherSpec = TEACHER_LIBERO) -> dict:
    t = teacher_macs(spec)
    s = student_macs(cfg)
    ratio = t["total"] / s["total"]
    return {
        "teacher": t,
        "student": s,
        "student_config": asdict(cfg),
        "compression_x": round(ratio, 1),
        "meets_100x": ratio >= 100.0,
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Teacher vs student FLOP profiler.")
    d = StudentConfig()
    p.add_argument("--variant", default=d.variant, choices=["hybrid", "cnn", "tinyvit"])
    p.add_argument("--dim", type=int, default=d.dim)
    p.add_argument("--num-conv-blocks", type=int, default=d.num_conv_blocks)
    p.add_argument("--conv-kernel", type=int, default=d.conv_kernel)
    p.add_argument("--num-attn-blocks", type=int, default=d.num_attn_blocks)
    p.add_argument("--attn-heads", type=int, default=d.attn_heads)
    p.add_argument("--attn-mlp-ratio", type=float, default=d.attn_mlp_ratio)
    p.add_argument("--head-hidden", type=int, default=d.head_hidden)
    p.add_argument("--json", action="store_true", help="emit JSON only")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    cfg = StudentConfig(
        variant=args.variant,
        dim=args.dim,
        num_conv_blocks=args.num_conv_blocks,
        conv_kernel=args.conv_kernel,
        num_attn_blocks=args.num_attn_blocks,
        attn_heads=args.attn_heads,
        attn_mlp_ratio=args.attn_mlp_ratio,
        head_hidden=args.head_hidden,
    )
    rep = profile(cfg)
    if args.json:
        print(json.dumps(rep, indent=2))
        return
    t, s = rep["teacher"], rep["student"]
    print(f"Teacher {TEACHER_LIBERO.name}: {t['gflops']} GFLOPs/crop "
          f"(patch_embed={2*t['patch_embed']/1e9:.2f} + resblocks={2*t['resblocks']/1e9:.2f})")
    print(f"Student ({cfg.variant}, dim={cfg.dim}, conv={cfg.num_conv_blocks}, "
          f"attn={cfg.num_attn_blocks}): {s['gflops']} GFLOPs/crop")
    for k in ("stem", "conv_blocks", "attn_blocks", "head"):
        if k in s:
            print(f"    {k:12s} {2*s[k]/1e9:8.3f} GFLOPs")
    verdict = "OK  >=100x" if rep["meets_100x"] else "!!  <100x (tune down)"
    print(f"Compression: {rep['compression_x']}x   [{verdict}]")


if __name__ == "__main__":
    main()
