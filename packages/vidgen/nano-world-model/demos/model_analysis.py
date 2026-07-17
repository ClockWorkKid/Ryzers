#!/usr/bin/env python
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
Full NanoWM system breakdown on Strix Halo (gfx1151): per-component parameter count, GFLOPs,
measured latency and token counts across the whole video-generation pipeline
(SD-VAE encode -> NanoWM DiT sampling loop -> SD-VAE decode), rendered as a single annotated
system-diagram PNG plus a machine-readable analysis.json.

Runs against a downloaded checkpoint's config (default: dino_wm_pusht). No dataset needed.
"""
import os, sys, json, time, argparse
from collections import defaultdict

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, "/repos/nanowm")
sys.path.insert(0, "/repos/nanowm/src")
sys.path.insert(0, "/repos/nanowm/src/sample")  # sampling_utils (encode_frames/decode_latents)

from models import get_models
from latent_codecs import (
    get_model_latent_channels, get_model_latent_size,
    load_autoencoder_kl, resolve_latent_codec_config,
)

try:
    from torch.utils.flop_counter import FlopCounterMode
    _HAS_FLOP = True
except Exception:
    _HAS_FLOP = False


def human(n, unit=""):
    for s, d in [("T", 1e12), ("G", 1e9), ("M", 1e6), ("K", 1e3)]:
        if abs(n) >= d:
            return f"{n/d:.2f}{s}{unit}"
    return f"{n:.2f}{unit}"


def bucket_of(key: str):
    k = key.lower()
    if key.endswith(".attn"):                     return "attention"
    if key.endswith(".mlp"):                      return "mlp (SwiGLU)"
    if key.endswith("adaln_modulation"):          return "adaLN cond"
    if key.endswith("x_embedder"):                return "patch embed"
    if key.endswith("t_embedder"):                return "timestep embed"
    if key.endswith("action_embedder"):           return "action embed"
    if key.endswith("final_layer"):               return "final layer"
    return None


def param_breakdown(model):
    buckets = defaultdict(int)
    for name, p in model.named_parameters():
        n = p.numel()
        if ".attn" in name:                         buckets["attention"] += n
        elif ".mlp" in name:                        buckets["mlp (SwiGLU)"] += n
        elif "adaln_modulation" in name.lower():    buckets["adaLN cond"] += n
        elif "x_embedder" in name:                  buckets["patch embed"] += n
        elif "t_embedder" in name:                  buckets["timestep embed"] += n
        elif "action_embedder" in name:             buckets["action embed"] += n
        elif "final_layer" in name:                 buckets["final layer"] += n
        elif "pos_embed" in name or "temp_embed" in name: buckets["pos/temp embed"] += n
        else:                                        buckets["other"] += n
    return dict(buckets)


def flops_breakdown(model, inputs):
    if not _HAS_FLOP:
        return {}, 0
    fcm = FlopCounterMode(display=False, depth=None)
    with fcm:
        model(**inputs)
    counts = fcm.get_flop_counts()
    buckets = defaultdict(int)
    for key, opmap in counts.items():
        if key == "Global":
            continue
        b = bucket_of(key)
        if b is not None:
            buckets[b] += sum(opmap.values())
    total = sum(counts.get("Global", {}).values())
    return dict(buckets), total


def latency_breakdown(model, inputs, iters=10, warmup=3):
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

    for name, mod in model.named_modules():
        if name.endswith(".attn"):                     register(mod, "attention")
        elif name.endswith(".mlp"):                    register(mod, "mlp (SwiGLU)")
        elif name.endswith("adaLN_modulation"):        register(mod, "adaLN cond")
        elif name.endswith("x_embedder"):              register(mod, "patch embed")
        elif name.endswith("t_embedder"):              register(mod, "timestep embed")
        elif name.endswith("action_embedder"):         register(mod, "action embed")
        elif name.endswith("final_layer"):             register(mod, "final layer")

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
    # Sync after each warmup call so one-time MIOpen/hipRTC kernel compilation completes here
    # (esp. the SD-VAE decoder's large convs) instead of leaking into the timed region.
    for _ in range(warmup):
        fn()
        if dev == "cuda":
            torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if dev == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0  # ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="dino_wm_pusht")
    ap.add_argument("--results_dir", default=os.environ.get("RESULTS_DIR", "/models/results"))
    ap.add_argument("--out", default="/outputs/analysis")
    ap.add_argument("--batch", type=int, default=1)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ckpt_dir = os.path.join(args.results_dir, args.domain)
    cfg = OmegaConf.load(os.path.join(ckpt_dir, "config.yaml"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_grad_enabled(False)

    # The shipped DINO-WM configs don't record a latent_codec block; rollout.py injects the
    # SD-VAE path from --vae_model_path / VAE_MODEL_PATH. Mirror that so the codec resolves.
    vae_path = os.environ.get("VAE_MODEL_PATH", "stabilityai/sd-vae-ft-mse")
    OmegaConf.set_struct(cfg, False)
    OmegaConf.update(cfg, "latent_codec.kind", "sd_vae", force_add=True)
    OmegaConf.update(cfg, "latent_codec.model_path", vae_path, force_add=True)

    latent_size = get_model_latent_size(cfg)
    latent_ch = get_model_latent_channels(cfg)
    cfg.model.latent_size = latent_size
    model = get_models(cfg).to(device).eval()

    ckpt = os.path.join(ckpt_dir, "model.pt")
    if os.path.isfile(ckpt):
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = sd.get("model", sd)
        sd = {k[6:] if k.startswith("model.") else k: v for k, v in sd.items()}
        model.load_state_dict(sd, strict=False)

    B = args.batch
    T = int(cfg.model.num_frames)
    patch = model.x_embedder.patch_size[0]
    tokens_per_frame = (latent_size // patch) ** 2
    act_dim = int(cfg.dataset.spec.action_dim) * int(cfg.dataset.frame_interval)

    x = torch.randn(B, T, latent_ch, latent_size, latent_size, device=device)
    t = torch.randint(0, int(cfg.experiment.diffusion.diffusion_steps), (B, T), device=device)
    action = torch.randn(B, T, act_dim, device=device) if cfg.model.use_action else None
    inp = dict(x=x, t=t, y=None, action=action, use_fp16=False)

    # ---- Tensor shapes at each pipeline stage + factorized-attention detail ----
    # NanoWM is Latte-style: alternating SPATIAL self-attn (seq=P patches/frame) and causal TEMPORAL
    # self-attn (seq=F frames/patch), depth/2 pairs each. Attention is O(P^2)+O(F^2), not O((P*F)^2).
    hidden = int(model.hidden_size)
    depth = len(model.blocks)
    heads = int(getattr(model.blocks[0].attn, "num_heads", 0)) or None
    img = int(cfg.model.image_size)
    tensor_shapes = {
        "input_frames_BFCHW": [B, T, 3, img, img],
        "vae_latents_BFCHW": [B, T, latent_ch, latent_size, latent_size],
        "patch_tokens_spatial_(B*F,P,D)": [B * T, tokens_per_frame, hidden],
        "temporal_tokens_(B*P,F,D)": [B * tokens_per_frame, T, hidden],
        "action_in_BFA": [B, T, act_dim],
        "adaLN_cond_per_block_(6D)": 6 * hidden,
        "output_BFCHW": [B, T, latent_ch, latent_size, latent_size],
    }
    attention_detail = {
        "kind": "factorized (Latte): spatial + causal-temporal",
        "spatial_seq_len_P": int(tokens_per_frame),
        "temporal_seq_len_F": int(T),
        "num_spatial_blocks": depth // 2,
        "num_temporal_blocks": depth - depth // 2,
        "num_heads": heads,
        "head_dim": (hidden // heads) if heads else None,
        "temporal_causal": bool(getattr(model, "causal", True)),
        "note": ("spatial attn seq=P dominates; temporal seq=F is tiny -> attention is far from the "
                 "O((P*F)^2) full-spatiotemporal cost"),
    }

    print("Params breakdown...")
    params = param_breakdown(model)
    total_params = sum(p.numel() for p in model.parameters())

    print("FLOPs breakdown (one DiT forward)...")
    flops, total_flops = flops_breakdown(model, inp)

    print("Latency breakdown (per DiT component)...")
    lat = latency_breakdown(model, inp) if device == "cuda" else {}
    dit_ms = time_module(lambda: model(**inp))

    # ---- Precision sweep: the released rollout runs the DiT in fp32. gfx1151 has fast packed
    # fp16/bf16 paths, so autocast is a prime speedup lever. Measure the same forward under autocast
    # (params stay fp32; matmul/conv compute in low precision) to estimate the achievable speedup. ----
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

    # ---- VAE (SD-VAE) encode/decode ----
    # The SD-VAE is identical across all NanoWM variants (same sd-vae-ft-mse, image 256 -> latent 32).
    # ANALYSIS_SKIP_VAE=1 skips re-timing it (measure once, reuse in the cross-variant aggregation).
    vae_stats = {}
    if os.environ.get("ANALYSIS_SKIP_VAE") == "1":
        print("VAE encode/decode... SKIPPED (shared SD-VAE; timed once elsewhere)")
    else:
      print("VAE encode/decode...")
      try:
        from sampling_utils import encode_frames, decode_latents
        vae_prec = getattr(getattr(cfg.experiment, "infra", object()), "vae_precision", "fp32")
        codec = resolve_latent_codec_config(cfg)
        vae = load_autoencoder_kl(codec.model_path).to(device).eval()
        # Time at the SAME batch the rollout decodes at (B_roll*T frames in one call). MIOpen's
        # conv-algo choice for the SD-VAE decoder is strongly batch-dependent on gfx1151: a small
        # batch (~16) selects a very slow fallback (~8 s/frame), while the rollout's larger batch
        # (~64) hits a fast algo (~0.4 s/frame). Matching the rollout batch makes the reported
        # per-frame latency representative of real generation. Upstream default precision is fp32.
        F_time = int(os.environ.get("ANALYSIS_VAE_BATCH", "64"))
        img = torch.randn(1, F_time, 3, cfg.model.image_size, cfg.model.image_size, device=device)
        lat_in = torch.randn(1, F_time, latent_ch, latent_size, latent_size, device=device)
        enc_ms = time_module(lambda: encode_frames(vae, img, device, vae_precision=vae_prec),
                             iters=3, warmup=2) / F_time
        dec_ms = time_module(lambda: decode_latents(vae, lat_in, vae_precision=vae_prec),
                             iters=3, warmup=2) / F_time
        vae_params = sum(p.numel() for p in vae.parameters())
        enc_flops = dec_flops = 0
        if _HAS_FLOP:
            img4 = torch.randn(1, 3, cfg.model.image_size, cfg.model.image_size, device=device)
            lat4 = torch.randn(1, latent_ch, latent_size, latent_size, device=device)
            f = FlopCounterMode(display=False)
            with f: vae.encode(img4).latent_dist.sample()
            enc_flops = sum(f.get_flop_counts().get("Global", {}).values())
            f = FlopCounterMode(display=False)
            with f: vae.decode(lat4).sample
            dec_flops = sum(f.get_flop_counts().get("Global", {}).values())
        vae_stats = dict(params=vae_params, encode_ms=enc_ms, decode_ms=dec_ms,
                         encode_flops=enc_flops, decode_flops=dec_flops, note="ms are per-frame")
      except Exception as e:
        print("VAE analysis skipped:", e)

    # ---- Full-rollout cost model ----
    # The DINO-WM rollout demos sample with 50 DDIM steps (rollout.py --num_sampling_steps default),
    # not the model-config's training default; use the demo value so the estimate matches the
    # measured per-sample wall-clock (~107 s/sample). Override with ANALYSIS_SAMPLING_STEPS.
    S = int(os.environ.get("ANALYSIS_SAMPLING_STEPS", "50"))
    gen_frames = 16
    rollout = dict(
        num_sampling_steps=S, generated_frames=gen_frames,
        model_cfg_sampling_steps=int(getattr(cfg.model, "num_sampling_steps", S)),
        dit_forward_ms=dit_ms,
        est_sampling_ms=dit_ms * S * gen_frames,
        vae_encode_ms=vae_stats.get("encode_ms", 0) * gen_frames,
        vae_decode_ms=vae_stats.get("decode_ms", 0) * gen_frames,
    )

    result = dict(
        domain=args.domain, device="gfx1151 (Radeon 8060S)",
        arch=str(cfg.model.arch), hidden_size=int(model.hidden_size),
        depth=len(model.blocks), num_frames=T, patch_size=int(patch),
        latent=f"{latent_ch}x{latent_size}x{latent_size}",
        tokens_per_frame_spatial=int(tokens_per_frame),
        temporal_len=int(T),
        total_spatiotemporal_tokens=int(tokens_per_frame * T),
        action_dim=int(act_dim),
        total_params=int(total_params),
        params_by_component=params,
        dit_gflops_per_forward=total_flops / 1e9,
        gflops_by_component={k: v / 1e9 for k, v in flops.items()},
        latency_ms_by_component=lat,
        dit_forward_ms=dit_ms,
        precision_sweep_ms=precision_sweep,
        tensor_shapes=tensor_shapes,
        attention_detail=attention_detail,
        num_heads=heads,
        vae=vae_stats,
        rollout_cost=rollout,
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
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], hspace=0.28, wspace=0.18)
    ax = fig.add_subplot(gs[0, :]); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 30)

    vae = r.get("vae", {})
    comp = r["params_by_component"]; fl = r["gflops_by_component"]; la = r["latency_ms_by_component"]
    dit_params = sum(v for k, v in comp.items())

    def box(x, w, title, lines, color):
        ax.add_patch(FancyBboxPatch((x, 9), w, 12, boxstyle="round,pad=0.3",
                                    fc=color, ec="#333", lw=1.5))
        ax.text(x + w / 2, 19.4, title, ha="center", va="top", fontsize=11, fontweight="bold")
        ax.text(x + w / 2, 16.6, "\n".join(lines), ha="center", va="top", fontsize=8.5)

    def arrow(x0, x1, label=""):
        ax.add_patch(FancyArrowPatch((x0, 15), (x1, 15), arrowstyle="-|>",
                                     mutation_scale=18, lw=1.6, color="#444"))
        if label:
            ax.text((x0 + x1) / 2, 22, label, ha="center", fontsize=8, color="#0a5")

    box(1, 15, "Input frames",
        [f"{r['num_frames']} x 3 x {r['latent'].split('x')[-1]}²", "RGB context"], "#eef3fb")
    arrow(16, 20)
    box(20, 17, "SD-VAE Encoder",
        [f"params {human(vae.get('params',0))}", f"{vae.get('encode_ms',0):.1f} ms/frame",
         f"{human(vae.get('encode_flops',0),'FLOP')}"], "#e8f6ec")
    arrow(37, 41, "latents")
    box(41, 20, f"NanoWM DiT  ({r['arch']})",
        [f"depth {r['depth']} · d={r['hidden_size']} · patch {r['patch_size']}",
         f"params {human(dit_params)} · {r['dit_gflops_per_forward']:.1f} GFLOP/fwd",
         f"{r['total_spatiotemporal_tokens']} tokens · {r['dit_forward_ms']:.1f} ms/fwd",
         f"x  {r['rollout_cost']['num_sampling_steps']} DDIM steps  x  frames"], "#fdf0e6")
    arrow(61, 65, "pred")
    box(65, 17, "SD-VAE Decoder",
        [f"params {human(vae.get('params',0))}", f"{vae.get('decode_ms',0):.1f} ms/frame",
         f"{human(vae.get('decode_flops',0),'FLOP')}"], "#e8f6ec")
    arrow(82, 86)
    box(86, 13, "Video out", [f"gen / gt /", "compare mp4"], "#eef3fb")
    ax.text(50, 27, "NanoWM video-generation pipeline — per-component breakdown (Strix Halo gfx1151)",
            ha="center", fontsize=13, fontweight="bold")
    ax.text(51, 6.5,
            f"Full rollout ≈ VAE encode + ({r['dit_forward_ms']:.1f} ms × "
            f"{r['rollout_cost']['num_sampling_steps']} steps × generated frames) + VAE decode",
            ha="center", fontsize=9, style="italic", color="#555")

    # bar: DiT params
    ax2 = fig.add_subplot(gs[1, 0])
    items = sorted(comp.items(), key=lambda kv: -kv[1])
    ax2.barh([k for k, _ in items][::-1], [v / 1e6 for _, v in items][::-1], color="#c9743a")
    ax2.set_xlabel("Parameters (M)"); ax2.set_title("DiT parameters by component", fontsize=11)

    # bar: latency + gflops
    ax3 = fig.add_subplot(gs[1, 1])
    if la:
        its = sorted(la.items(), key=lambda kv: -kv[1])
        ax3.barh([k for k, _ in its][::-1], [v for _, v in its][::-1], color="#3a76c9")
        ax3.set_xlabel("Latency (ms) per DiT forward")
        ax3.set_title("DiT latency by component (measured on gfx1151)", fontsize=11)
    else:
        its = sorted(fl.items(), key=lambda kv: -kv[1])
        ax3.barh([k for k, _ in its][::-1], [v for _, v in its][::-1], color="#3a76c9")
        ax3.set_xlabel("GFLOPs per DiT forward")
        ax3.set_title("DiT GFLOPs by component", fontsize=11)

    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
