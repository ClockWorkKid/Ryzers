#!/usr/bin/env python
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
Latte system breakdown on Strix Halo (gfx1151): per-component parameter count, GFLOPs, measured
latency and token counts across the class-conditional video-generation pipeline
(SD-VAE encode -> Latte DiT denoising loop -> SD-VAE decode), rendered as an annotated
system-diagram PNG plus a machine-readable analysis.json.

Latte is a factorized-attention DiT: `blocks` alternate SPATIAL self-attn (seq = num_patches)
and TEMPORAL self-attn (seq = num_frames); one DiT forward denoises ALL frames jointly, so the
sampling cost is dit_forward_ms x num_sampling_steps (not x frames like an autoregressive WM).

Runs against a downloaded class checkpoint's config (default: sky). Needs /models/Latte + vae.
Env: DATASET (ffs|sky|taichi|ucf101), LATTE_DIR, ANALYSIS_SAMPLING_STEPS, ANALYSIS_VAE_BATCH.
"""
import os, sys, json, time, argparse
from collections import defaultdict

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, "/repos/latte")
from models import get_models
from utils import find_model
from diffusers.models import AutoencoderKL

try:
    from torch.utils.flop_counter import FlopCounterMode
    _HAS_FLOP = True
except Exception:
    _HAS_FLOP = False

CKPTS = {"ffs": "ffs.pt", "sky": "skytimelapse.pt", "taichi": "taichi-hd.pt", "ucf101": "ucf101.pt"}


def human(n, unit=""):
    for s, d in [("T", 1e12), ("G", 1e9), ("M", 1e6), ("K", 1e3)]:
        if abs(n) >= d:
            return f"{n/d:.2f}{s}{unit}"
    return f"{n:.2f}{unit}"


def param_breakdown(model):
    buckets = defaultdict(int)
    for name, p in model.named_parameters():
        if ".attn" in name:                         buckets["attention"] += p.numel()
        elif ".mlp" in name:                        buckets["mlp"] += p.numel()
        elif "adaLN_modulation" in name.lower():    buckets["adaLN cond"] += p.numel()
        elif "x_embedder" in name:                  buckets["patch embed"] += p.numel()
        elif "t_embedder" in name:                  buckets["timestep embed"] += p.numel()
        elif "y_embedder" in name:                  buckets["class embed"] += p.numel()
        elif "final_layer" in name:                 buckets["final layer"] += p.numel()
        elif "pos_embed" in name or "temp_embed" in name: buckets["pos/temp embed"] += p.numel()
        else:                                        buckets["other"] += p.numel()
    return dict(buckets)


def flops_total(model, inputs):
    if not _HAS_FLOP:
        return 0
    fcm = FlopCounterMode(display=False, depth=None)
    with fcm:
        model(**inputs)
    return sum(fcm.get_flop_counts().get("Global", {}).values())


def latency_by_kind(model, inputs, iters=10, warmup=3):
    """Attn/MLP latency split into spatial (even block) vs temporal (odd block) by index parity."""
    cats = {}
    handles = []
    pending = []

    def register(mod, cat):
        def pre(m, i):
            s = torch.cuda.Event(enable_timing=True); s.record(); m._t0 = s
        def post(m, i, o):
            e = torch.cuda.Event(enable_timing=True); e.record(); pending.append((cat, m._t0, e))
        handles.append(mod.register_forward_pre_hook(pre))
        handles.append(mod.register_forward_hook(post))

    for idx, block in enumerate(model.blocks):
        kind = "spatial" if idx % 2 == 0 else "temporal"
        register(block.attn, f"attn ({kind})")
        register(block.mlp, f"mlp ({kind})")
    if hasattr(model, "final_layer"):
        register(model.final_layer, "final layer")

    dev = next(model.parameters()).device
    for it in range(warmup + iters):
        pending.clear()
        with torch.no_grad():
            model(**inputs)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        if it >= warmup:
            for cat, s, e in pending:
                cats[cat] = cats.get(cat, 0.0) + s.elapsed_time(e)
    for h in handles:
        h.remove()
    return {k: v / iters for k, v in cats.items()}


def time_module(fn, iters=10, warmup=5):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for _ in range(warmup):
        fn()
        if dev == "cuda":
            torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if dev == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=os.environ.get("DATASET", "sky"))
    ap.add_argument("--latte_dir", default=os.environ.get("LATTE_DIR", "/models/Latte"))
    ap.add_argument("--out", default="/outputs/analysis")
    ap.add_argument("--batch", type=int, default=1)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_grad_enabled(False)

    cfg = OmegaConf.load(f"/repos/latte/configs/{args.dataset}/{args.dataset}_sample.yaml")
    latent = int(cfg.image_size) // 8
    model = get_models(OmegaConf.create({**cfg, "latent_size": latent})).to(device).eval()
    sd = find_model(os.path.join(args.latte_dir, CKPTS[args.dataset]))
    model.load_state_dict(sd)
    for m in model.modules():
        if hasattr(m, "attention_mode"):
            m.attention_mode = "flash"  # ROCm SDPA fast path

    B = args.batch
    T = int(cfg.num_frames)
    patch = model.x_embedder.patch_size[0]
    tokens_per_frame = (latent // patch) ** 2
    hidden = int(model.hidden_size)
    depth = len(model.blocks)
    heads = int(getattr(model.blocks[0].attn, "num_heads", 0)) or None

    extras = int(cfg.get("extras", 1) or 1)
    nc = cfg.get("num_classes", None)
    nc = int(nc) if nc else 0
    x = torch.randn(B, T, 4, latent, latent, device=device)
    t = torch.randint(0, 1000, (B,), device=device)
    y = torch.randint(0, nc, (B,), device=device) if (extras == 2 and nc) else None
    inp = dict(x=x, t=t, y=y)

    tensor_shapes = {
        "input_frames_BFCHW": [B, T, 3, int(cfg.image_size), int(cfg.image_size)],
        "vae_latents_BFCHW": [B, T, 4, latent, latent],
        "spatial_tokens_(B*F,P,D)": [B * T, tokens_per_frame, hidden],
        "temporal_tokens_(B*P,F,D)": [B * tokens_per_frame, T, hidden],
        "adaLN_cond_per_block_(6D)": 6 * hidden,
        "output_BFCHW": [B, T, 4, latent, latent],
    }
    attention_detail = {
        "kind": "factorized (Latte): alternating spatial + temporal self-attn",
        "spatial_seq_len_P": int(tokens_per_frame),
        "temporal_seq_len_F": int(T),
        "num_spatial_blocks": depth // 2,
        "num_temporal_blocks": depth // 2,
        "num_heads": heads,
        "head_dim": (hidden // heads) if heads else None,
        "note": ("spatial attn (seq=P) dominates; temporal seq=F is tiny -> attention is far from "
                 "the O((P*F)^2) full-spatiotemporal cost"),
    }

    print("Params breakdown...")
    params = param_breakdown(model)
    total_params = sum(p.numel() for p in model.parameters())

    print("FLOPs (one DiT forward)...")
    total_flops = flops_total(model, inp)

    print("Latency breakdown (per DiT component)...")
    lat = latency_by_kind(model, inp) if device == "cuda" else {}
    dit_ms = time_module(lambda: model(**inp))

    print("Precision sweep (autocast; params stay fp32)...")
    precision_sweep = {"fp32": round(dit_ms, 2)}
    if device == "cuda":
        for name, dt in (("fp16", torch.float16), ("bf16", torch.bfloat16)):
            try:
                def run_ac(dt=dt):
                    with torch.autocast("cuda", dtype=dt):
                        model(**inp)
                precision_sweep[name] = round(time_module(run_ac), 2)
            except Exception as e:
                precision_sweep[name] = f"failed: {type(e).__name__}: {e}"

    # SD-VAE timing. Generation runs the VAE in fp16 (see demos/_gen_frames); fp32 decode hits a
    # pathologically slow MIOpen conv fallback on gfx1151 (>1 min/16-frame batch), so we time the
    # fp16 path that real sampling uses. Set ANALYSIS_VAE_FP32=1 to also probe fp32 (very slow).
    print("VAE encode/decode (fp16)...", flush=True)
    vae_stats = {}
    try:
        vae = AutoencoderKL.from_pretrained(os.path.join(args.latte_dir, "vae")).to(device).eval().half()
        Fb = int(os.environ.get("ANALYSIS_VAE_BATCH", "8"))
        img = torch.randn(Fb, 3, int(cfg.image_size), int(cfg.image_size), device=device, dtype=torch.float16)
        lat_in = torch.randn(Fb, 4, latent, latent, device=device, dtype=torch.float16)
        enc_ms = time_module(lambda: vae.encode(img).latent_dist.sample(), iters=2, warmup=1) / Fb
        dec_ms = time_module(lambda: vae.decode(lat_in / 0.18215).sample, iters=2, warmup=1) / Fb
        vae_params = sum(p.numel() for p in vae.parameters())
        enc_flops = dec_flops = 0
        if _HAS_FLOP:
            i1 = torch.randn(1, 3, int(cfg.image_size), int(cfg.image_size), device=device, dtype=torch.float16)
            l1 = torch.randn(1, 4, latent, latent, device=device, dtype=torch.float16)
            f = FlopCounterMode(display=False)
            with f: vae.encode(i1).latent_dist.sample()
            enc_flops = sum(f.get_flop_counts().get("Global", {}).values())
            f = FlopCounterMode(display=False)
            with f: vae.decode(l1).sample
            dec_flops = sum(f.get_flop_counts().get("Global", {}).values())
        vae_stats = dict(params=vae_params, precision="fp16", encode_ms=enc_ms, decode_ms=dec_ms,
                         encode_flops=enc_flops, decode_flops=dec_flops, note="ms are per-frame, fp16")
    except Exception as e:
        print("VAE analysis skipped:", e)

    S = int(os.environ.get("ANALYSIS_SAMPLING_STEPS", "50"))
    rollout = dict(
        num_sampling_steps=S, generated_frames=T,
        dit_forward_ms=dit_ms,
        est_sampling_ms=dit_ms * S,  # DiT denoises all frames jointly per step
        vae_decode_ms=vae_stats.get("decode_ms", 0) * T,
        note="Latte denoises all frames jointly: sampling = dit_forward_ms x steps (not x frames)",
    )

    result = dict(
        dataset=args.dataset, device="gfx1151 (Radeon 8060S)",
        arch="Latte-XL/2", hidden_size=hidden, depth=depth,
        num_frames=T, patch_size=int(patch), latent=f"4x{latent}x{latent}",
        image_size=int(cfg.image_size),
        spatial_tokens_per_frame=int(tokens_per_frame), temporal_len=int(T),
        total_params=int(total_params), params_by_component=params,
        dit_gflops_per_forward=total_flops / 1e9,
        latency_ms_by_component=lat, dit_forward_ms=dit_ms,
        precision_sweep_ms=precision_sweep,
        tensor_shapes=tensor_shapes, attention_detail=attention_detail,
        num_heads=heads, vae=vae_stats, rollout_cost=rollout,
    )
    with open(os.path.join(args.out, "analysis.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    render_diagram(result, os.path.join(args.out, "system_diagram.png"))
    print("wrote", os.path.join(args.out, "system_diagram.png"))


def render_diagram(r, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], hspace=0.3, wspace=0.18)
    ax = fig.add_subplot(gs[0, :]); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 30)

    vae = r.get("vae", {})
    comp = r["params_by_component"]; la = r["latency_ms_by_component"]
    dit_params = sum(comp.values())

    def box(x, w, title, lines, color):
        ax.add_patch(FancyBboxPatch((x, 9), w, 12, boxstyle="round,pad=0.3", fc=color, ec="#333", lw=1.5))
        ax.text(x + w / 2, 19.4, title, ha="center", va="top", fontsize=11, fontweight="bold")
        ax.text(x + w / 2, 16.6, "\n".join(lines), ha="center", va="top", fontsize=8.5)

    def arrow(x0, x1, label=""):
        ax.add_patch(FancyArrowPatch((x0, 15), (x1, 15), arrowstyle="-|>", mutation_scale=18, lw=1.6, color="#444"))
        if label:
            ax.text((x0 + x1) / 2, 22, label, ha="center", fontsize=8, color="#0a5")

    box(1, 15, "Input frames", [f"{r['num_frames']} x 3 x {r['image_size']}²", "RGB"], "#eef3fb")
    arrow(16, 20)
    box(20, 17, "SD-VAE Encoder",
        [f"params {human(vae.get('params',0))}", f"{vae.get('encode_ms',0):.1f} ms/frame",
         f"{human(vae.get('encode_flops',0),'FLOP')}"], "#e8f6ec")
    arrow(37, 41, "latents")
    box(41, 20, f"Latte DiT ({r['arch']})",
        [f"depth {r['depth']} · d={r['hidden_size']} · patch {r['patch_size']}",
         f"params {human(dit_params)} · {r['dit_gflops_per_forward']:.1f} GFLOP/fwd",
         f"P={r['spatial_tokens_per_frame']} · F={r['temporal_len']} · {r['dit_forward_ms']:.1f} ms/fwd",
         f"x {r['rollout_cost']['num_sampling_steps']} steps (joint frames)"], "#fdf0e6")
    arrow(61, 65, "pred")
    box(65, 17, "SD-VAE Decoder",
        [f"params {human(vae.get('params',0))}", f"{vae.get('decode_ms',0):.1f} ms/frame",
         f"{human(vae.get('decode_flops',0),'FLOP')}"], "#e8f6ec")
    arrow(82, 86)
    box(86, 13, "Video out", ["mp4 /", "gif"], "#eef3fb")
    ax.text(50, 27, "Latte video-generation pipeline — per-component breakdown (Strix Halo gfx1151)",
            ha="center", fontsize=13, fontweight="bold")
    ax.text(51, 6.5,
            f"Full sample ≈ VAE encode + ({r['dit_forward_ms']:.1f} ms × "
            f"{r['rollout_cost']['num_sampling_steps']} steps, all frames joint) + VAE decode × {r['num_frames']}",
            ha="center", fontsize=9, style="italic", color="#555")

    ax2 = fig.add_subplot(gs[1, 0])
    items = sorted(comp.items(), key=lambda kv: -kv[1])
    ax2.barh([k for k, _ in items][::-1], [v / 1e6 for _, v in items][::-1], color="#c9743a")
    ax2.set_xlabel("Parameters (M)"); ax2.set_title("DiT parameters by component", fontsize=11)

    ax3 = fig.add_subplot(gs[1, 1])
    if la:
        its = sorted(la.items(), key=lambda kv: -kv[1])
        ax3.barh([k for k, _ in its][::-1], [v for _, v in its][::-1], color="#3a76c9")
        ax3.set_xlabel("Latency (ms) per DiT forward")
        ax3.set_title("DiT latency by component (measured, gfx1151)", fontsize=11)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
