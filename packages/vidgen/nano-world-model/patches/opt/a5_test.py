#!/usr/bin/env python
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
A5 (context-frame KV cache) exactness + latency test. For a frozen context window and a varying
target frame, the A5 prime/use path must reproduce the non-A5 full-window forward's TARGET-frame
output bit-closely (fp reduction-order noise only). Also times a per-step 'use' forward vs a full
forward (the amortized rollout gain, since context is primed once per window).
"""
import os, sys, time, argparse, importlib.util, contextlib
import torch
from omegaconf import OmegaConf

sys.path.insert(0, "/repos/nanowm"); sys.path.insert(0, "/repos/nanowm/src")
from latent_codecs import get_model_latent_channels, get_model_latent_size


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def build(mod, cfg, ls, lc):
    kw = dict(input_size=ls, in_channels=lc,
              num_classes=int(getattr(cfg.model, "num_classes", 1000)),
              num_frames=int(cfg.model.num_frames), extras=int(getattr(cfg.model, "extras", 1)),
              use_action=bool(cfg.model.use_action),
              action_dim=int(cfg.dataset.spec.action_dim) * int(cfg.dataset.frame_interval),
              action_injection_type=str(cfg.model.action_injection.type),
              causal=bool(getattr(cfg.model, "causal", True)))
    torch.manual_seed(0)
    return mod.NanoWM_models[str(cfg.model.arch)](**kw).cuda().eval()


def tgt(out, n_ctx):
    return out[:, n_ctx:].float()


def timed(fn, iters=12, warmup=4):
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1000.0


def run_domain(domain, results_dir, opt_path):
    cfg = OmegaConf.load(os.path.join(results_dir, domain, "config.yaml")); OmegaConf.set_struct(cfg, False)
    ls = get_model_latent_size(cfg); lc = get_model_latent_channels(cfg)
    mod = load_module("opt_nanowm", opt_path)
    model = build(mod, cfg, ls, lc)
    sd = torch.load(os.path.join(results_dir, domain, "model.pt"), map_location="cpu", weights_only=False)
    sd = sd.get("model", sd); sd = {k[6:] if k.startswith("model.") else k: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)

    if not bool(getattr(cfg.model, "causal", True)):
        print(f"\n### {domain}: non-causal model -> A5 not applicable; skipping"); return

    F = int(cfg.model.num_frames)
    n_ctx = max(1, min(int(getattr(cfg.model, "n_context_frames", 1) or 1), F - 1))
    Fw = n_ctx + 1                       # trimmed window (context + 1 target)
    act = int(cfg.dataset.spec.action_dim) * int(cfg.dataset.frame_interval)
    steps = int(cfg.experiment.diffusion.diffusion_steps)
    t_stab = int(round(0.02 * (steps - 1)))
    print(f"\n==================== {domain}  arch={cfg.model.arch}  n_ctx={n_ctx}  window={Fw} ====================")

    torch.manual_seed(7)
    ctx = torch.randn(1, n_ctx, lc, ls, ls, device="cuda")
    a = torch.randn(1, Fw, act, device="cuda") if cfg.model.use_action else None
    tt = torch.full((1, Fw), 0, device="cuda", dtype=torch.long); tt[:, :n_ctx] = t_stab
    tt[:, n_ctx:] = steps // 2  # some mid target noise level

    def win(target):  # [1, Fw, ...]
        return torch.cat([ctx, target], dim=1)

    for prec_name, dt in (("fp32", None), ("bf16", torch.bfloat16)):
        ctxm = torch.autocast("cuda", dtype=dt) if dt is not None else contextlib.nullcontext()
        with torch.no_grad(), ctxm:
            tgt1 = torch.randn(1, 1, lc, ls, ls, device="cuda")
            tgt2 = torch.randn(1, 1, lc, ls, ls, device="cuda")
            # reference: non-A5 full-window forward
            model.a5_configure(0, enabled=False)
            ref1 = tgt(model(win(tgt1), tt, action=a), n_ctx)
            ref2 = tgt(model(win(tgt2), tt, action=a), n_ctx)
            # A5: prime on window-1 (same context), then 'use' on window-2 (same ctx, new target)
            model.a5_configure(n_ctx, enabled=True)
            prime1 = tgt(model(win(tgt1), tt, action=a), n_ctx)   # prime
            use2 = tgt(model(win(tgt2), tt, action=a), n_ctx)     # use (cached ctx)
            e_prime = (prime1 - ref1).abs().max().item()
            e_use = (use2 - ref2).abs().max().item()
            # new context -> must re-prime and stay correct
            ctx2 = torch.randn(1, n_ctx, lc, ls, ls, device="cuda")
            ctx_save = ctx
            def win2(target):
                return torch.cat([ctx2, target], dim=1)
            model.a5_configure(0, enabled=False); refB = tgt(model(win2(tgt1), tt, action=a), n_ctx)
            model.a5_configure(n_ctx, enabled=True)
            _ = model(win2(tgt1), tt, action=a)                    # re-prime on new ctx
            useB = tgt(model(win2(tgt2), tt, action=a), n_ctx)
            model.a5_configure(0, enabled=False); refB2 = tgt(model(win2(tgt2), tt, action=a), n_ctx)
            e_reprime = (useB - refB2).abs().max().item()

            # latency: full forward vs A5 'use' (steady-state per-step)
            model.a5_configure(0, enabled=False)
            ms_full = timed(lambda: model(win(tgt2), tt, action=a))
            model.a5_configure(n_ctx, enabled=True); model(win(tgt1), tt, action=a)  # prime once
            ms_use = timed(lambda: model(win(tgt2), tt, action=a))
        print(f"  {prec_name}: prime_vs_full={e_prime:.2e}  use_vs_full={e_use:.2e}  reprime_vs_full={e_reprime:.2e}"
              f"  | full={ms_full:7.1f}ms use={ms_use:7.1f}ms ({ms_full/ms_use:4.2f}x)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="+", default=["dino_wm_pusht", "csgo"])
    ap.add_argument("--results_dir", default="/models/results")
    ap.add_argument("--opt", default="/repos/nanowm/src/models/nanowm.py")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    for d in args.domains:
        run_domain(d, args.results_dir, args.opt)


if __name__ == "__main__":
    raise SystemExit(main())
