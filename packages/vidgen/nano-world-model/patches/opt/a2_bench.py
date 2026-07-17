#!/usr/bin/env python
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
A2 microbenchmark + exactness check. Builds the optimized NanoWM for a domain, loads real weights,
and times a single DiT forward with A2 (temporal adaLN de-dup) ON vs OFF, across precision
(fp32 / bf16) and window (trimmed history+1 vs full num_frames). Also asserts A2-on == A2-off
output (exactness). Single-forward latency scales directly to the 50-step sampling loop.
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


def timed(model, x, t, action, dtype, iters=12, warmup=4):
    ctx = torch.autocast("cuda", dtype=dtype) if dtype is not None else contextlib.nullcontext()
    with torch.no_grad(), ctx:
        for _ in range(warmup):
            _ = model(x, t, action=action)
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(iters):
            out = model(x, t, action=action)
        torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1000.0, out


def run_domain(domain, results_dir, opt_path):
    cfg = OmegaConf.load(os.path.join(results_dir, domain, "config.yaml")); OmegaConf.set_struct(cfg, False)
    ls = get_model_latent_size(cfg); lc = get_model_latent_channels(cfg)
    mod = load_module("opt_nanowm", opt_path)
    model = build(mod, cfg, ls, lc)
    sd = torch.load(os.path.join(results_dir, domain, "model.pt"), map_location="cpu", weights_only=False)
    sd = sd.get("model", sd); sd = {k[6:] if k.startswith("model.") else k: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)

    F = int(cfg.model.num_frames)
    n_ctx = int(getattr(cfg.model, "n_context_frames", 1) or 1)
    h1 = max(1, min(n_ctx, F - 1)) + 1  # trimmed window = history + 1
    act = int(cfg.dataset.spec.action_dim) * int(cfg.dataset.frame_interval)
    steps = int(cfg.experiment.diffusion.diffusion_steps)

    print(f"\n==================== {domain}  arch={cfg.model.arch}  full_F={F}  trim_F={h1} ====================")
    for win_name, Fw in (("trim", h1), ("full", F)):
        torch.manual_seed(1234)
        x = torch.randn(1, Fw, lc, ls, ls, device="cuda")
        t = torch.randint(0, steps, (1, Fw), device="cuda")
        a = torch.randn(1, Fw, act, device="cuda") if cfg.model.use_action else None

        # exactness: A2 on vs off, fp32
        model._a2 = False; o_off = model(x, t, action=a)
        model._a2 = True;  o_on = model(x, t, action=a)
        max_abs = (o_on.float() - o_off.float()).abs().max().item()

        row = f"  [{win_name} F={Fw}] A2on-vs-A2off max_abs={max_abs:.2e}"
        for prec_name, dt in (("fp32", None), ("bf16", torch.bfloat16)):
            model._a2 = False; ms_off, _ = timed(model, x, t, a, dt)
            model._a2 = True;  ms_on, _ = timed(model, x, t, a, dt)
            row += f" | {prec_name}: off={ms_off:7.1f}ms on={ms_on:7.1f}ms ({ms_off/ms_on:4.2f}x)"
        print(row)


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
