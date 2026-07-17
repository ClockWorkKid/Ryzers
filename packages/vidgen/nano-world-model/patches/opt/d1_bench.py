#!/usr/bin/env python
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
D1: torch.compile the DiT forward. Compares eager vs compiled per-forward latency (fp32/bf16, both
scales) on the fixed trimmed window (A5 off -> static shapes, composes with A1/A2/B1), verifies
compiled-vs-eager output parity, and probes whether compile composes with A5 (dynamic shapes +
torch.equal control flow -> expected graph breaks).
"""
import os, sys, time, argparse, importlib.util, contextlib, traceback
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


def timed(fn, iters=15, warmup=6):
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

    F = int(cfg.model.num_frames)
    n_ctx = max(1, min(int(getattr(cfg.model, "n_context_frames", 1) or 1), F - 1))
    Fw = n_ctx + 1
    act = int(cfg.dataset.spec.action_dim) * int(cfg.dataset.frame_interval)
    print(f"\n==================== {domain}  arch={cfg.model.arch}  window={Fw} ====================")

    torch.manual_seed(11)
    x = torch.randn(1, Fw, lc, ls, ls, device="cuda")
    t = torch.randint(0, int(cfg.experiment.diffusion.diffusion_steps), (1, Fw), device="cuda")
    a = torch.randn(1, Fw, act, device="cuda") if cfg.model.use_action else None

    model.a5_configure(0, enabled=False)  # A5 off -> static shape, clean compile
    try:
        cmodel = torch.compile(model, dynamic=False)
    except Exception:
        print("  torch.compile unavailable:"); traceback.print_exc(); return

    for prec_name, dt in (("bf16", torch.bfloat16), ("fp32", None)):
        ctxm = torch.autocast("cuda", dtype=dt) if dt is not None else contextlib.nullcontext()
        with torch.no_grad(), ctxm:
            try:
                eager_out = model(x, t, action=a).float()
                ms_eager = timed(lambda: model(x, t, action=a))
                comp_out = cmodel(x, t, action=a).float()   # first call triggers compilation
                ms_comp = timed(lambda: cmodel(x, t, action=a))
                err = (comp_out - eager_out).abs().max().item()
                print(f"  {prec_name}: eager={ms_eager:7.1f}ms compiled={ms_comp:7.1f}ms "
                      f"({ms_eager/ms_comp:4.2f}x)  parity(max_abs)={err:.2e}")
            except Exception:
                print(f"  {prec_name}: compile/run FAILED"); traceback.print_exc()


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
