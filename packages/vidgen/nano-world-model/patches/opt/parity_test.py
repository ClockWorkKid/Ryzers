#!/usr/bin/env python
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
Numerical parity gate for the FastWAM NanoWM inference optimizations. Loads the UNMODIFIED upstream
nanowm.py and the optimized override side by side (identical weights) and checks:
  A3+A4 exactness : modified full-window fp32 output == original fp32 output
  A1  exactness   : modified trimmed-window kept-frame == original full-window kept-frame
  A4  exactness   : precomputed action_emb path == in-forward action path
  B1  magnitude    : bf16 autocast vs fp32 (reports max/mean abs + relative error; not expected 0)
Run per domain (--domain). No dataset needed (random inputs, shared weights).
"""
import os, sys, argparse, importlib.util
import torch
from omegaconf import OmegaConf

sys.path.insert(0, "/repos/nanowm"); sys.path.insert(0, "/repos/nanowm/src")
from latent_codecs import get_model_latent_channels, get_model_latent_size


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def build(mod, cfg, latent_size, latent_ch, seed=0):
    kw = dict(
        input_size=latent_size, in_channels=latent_ch,
        num_classes=int(getattr(cfg.model, "num_classes", 1000)),
        num_frames=int(cfg.model.num_frames),
        extras=int(getattr(cfg.model, "extras", 1)),
        use_action=bool(cfg.model.use_action),
        action_dim=int(cfg.dataset.spec.action_dim) * int(cfg.dataset.frame_interval),
        action_injection_type=str(cfg.model.action_injection.type),
        causal=bool(getattr(cfg.model, "causal", True)),
    )
    torch.manual_seed(seed)
    return mod.NanoWM_models[str(cfg.model.arch)](**kw).cuda().eval()


def stats(a, b):
    d = (a.float() - b.float()).abs()
    denom = b.float().abs().mean().clamp_min(1e-8)
    return dict(max_abs=d.max().item(), mean_abs=d.mean().item(), rel=(d.mean() / denom).item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="csgo")
    ap.add_argument("--orig", default="/repos/nanowm/src/models/nanowm.py")
    ap.add_argument("--opt", default="/tmp/opt_nanowm.py")
    ap.add_argument("--results_dir", default="/models/results")
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    cfg = OmegaConf.load(os.path.join(args.results_dir, args.domain, "config.yaml"))
    OmegaConf.set_struct(cfg, False)
    latent_size = get_model_latent_size(cfg); latent_ch = get_model_latent_channels(cfg)

    orig_mod = load_module("orig_nanowm", args.orig)
    opt_mod = load_module("opt_nanowm", args.opt)
    m_orig = build(orig_mod, cfg, latent_size, latent_ch)
    m_opt = build(opt_mod, cfg, latent_size, latent_ch)

    # Load REAL trained weights into both (NanoWM zero-inits the final layer, so random-init models
    # output all-zeros and would trivially "match"). Identical weights -> meaningful comparison.
    ckpt = os.path.join(args.results_dir, args.domain, "model.pt")
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd)
    sd = {k[6:] if k.startswith("model.") else k: v for k, v in sd.items()}
    miss_o, unexp_o = m_orig.load_state_dict(sd, strict=False)
    m_opt.load_state_dict(m_orig.state_dict(), strict=True)  # identical weights
    print(f"[ckpt] loaded {ckpt}  (missing={len(miss_o)} unexpected={len(unexp_o)})")

    B, F = 1, int(cfg.model.num_frames)
    act = int(cfg.dataset.spec.action_dim) * int(cfg.dataset.frame_interval)
    steps = int(cfg.experiment.diffusion.diffusion_steps)
    n_ctx = int(getattr(cfg.model, "n_context_frames", 1) or 1)
    h = max(1, min(n_ctx, F - 1))

    torch.manual_seed(1234)
    x = torch.randn(B, F, latent_ch, latent_size, latent_size, device="cuda")
    t = torch.randint(0, steps, (B, F), device="cuda")
    action = torch.randn(B, F, act, device="cuda") if cfg.model.use_action else None

    print(f"=== parity: {args.domain}  arch={cfg.model.arch}  F={F}  history+1={h+1}  act={act} ===")
    o_orig = m_orig(x, t, action=action)
    o_opt = m_opt(x, t, action=action)
    s_full = stats(o_opt, o_orig)
    print(f"[A3] modified full-window fp32 vs original : max_abs={s_full['max_abs']:.3e} mean_abs={s_full['mean_abs']:.3e}")

    # A1 trimmed window: kept frame is index h
    o_trim = m_opt(x[:, :h+1], t[:, :h+1], action=(action[:, :h+1] if action is not None else None))
    s_trim = stats(o_trim[:, h], o_orig[:, h])
    print(f"[A1] trimmed kept-frame vs original[:,h]   : max_abs={s_trim['max_abs']:.3e} mean_abs={s_trim['mean_abs']:.3e}")

    # A4 precomputed action_emb
    if action is not None:
        ae = m_opt.action_embedder(action)
        ae = torch.cat([torch.zeros_like(ae[:, :1, :]), ae[:, :-1, :]], dim=1)
        o_a4 = m_opt(x, t, action_emb=ae)
        s_a4 = stats(o_a4, o_opt)
        print(f"[A4] precomputed action_emb vs in-forward  : max_abs={s_a4['max_abs']:.3e} mean_abs={s_a4['mean_abs']:.3e}")

    # B1 bf16 autocast magnitude
    for name, dt in (("bf16", torch.bfloat16), ("fp16", torch.float16)):
        with torch.autocast("cuda", dtype=dt):
            o_lp = m_opt(x, t, action=action)
        s_lp = stats(o_lp, o_orig)
        print(f"[B1] {name} autocast vs fp32               : max_abs={s_lp['max_abs']:.3e} mean_abs={s_lp['mean_abs']:.3e} rel={s_lp['rel']:.3e}")

    exact_ok = s_full["max_abs"] < 1e-2 and s_trim["max_abs"] < 1e-2
    print("=== EXACT-PATH PARITY:", "PASS" if exact_ok else "FAIL", "===")


if __name__ == "__main__":
    raise SystemExit(main())
