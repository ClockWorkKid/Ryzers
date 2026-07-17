#!/usr/bin/env python
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
Full-stack cumulative waterfall bench. Stacks the exact/numerical optimizations one on top of the
next and reports, in ONE apples-to-apples run per domain:

  V0 baseline  : fp32, FULL window (num_frames), A2 off, A5 off   (upstream-equivalent per-step)
  +A1 (trim)   : fp32, TRIM window (history+1),  A2 off, A5 off
  +A2          : fp32, trim,                      A2 ON,  A5 off
  +B1 (bf16)   : bf16, trim,                      A2 on,  A5 off
  +A5          : bf16, trim,                      A2 on,  A5 ON (steady-state 'use')

For each layer: per-step DiT-forward latency, cumulative speedup vs V0, and marginal speedup vs the
previous layer. Then a per-generated-frame projection (sampling loop = S steps; A5 costs one prime +
(S-1) 'use' forwards) at the shipped S=50 and at the C1 recommended S, giving the overall stacked
speedup. Also reports the exactness of the full stack: fp32 target-frame vs the V0 upstream target
(should be fp reduction-order noise ~1e-6), and the bf16 full-stack target vs the fp32 full-stack
target (the single lossy delta, pure bf16 rounding). A3 (RoPE precompute) is baked into the
optimized forward; A4 (action-embed cache) is a per-step loop amortization, negligible per-forward.
"""
import os, sys, time, json, argparse, importlib.util, contextlib
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


def run_domain(domain, results_dir, opt_path, ship_steps, c1_steps):
    cfg = OmegaConf.load(os.path.join(results_dir, domain, "config.yaml")); OmegaConf.set_struct(cfg, False)
    ls = get_model_latent_size(cfg); lc = get_model_latent_channels(cfg)
    mod = load_module("opt_nanowm", opt_path)
    model = build(mod, cfg, ls, lc)
    sd = torch.load(os.path.join(results_dir, domain, "model.pt"), map_location="cpu", weights_only=False)
    sd = sd.get("model", sd); sd = {k[6:] if k.startswith("model.") else k: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)

    causal = bool(getattr(cfg.model, "causal", True))
    F = int(cfg.model.num_frames)
    n_ctx = max(1, min(int(getattr(cfg.model, "n_context_frames", 1) or 1), F - 1))
    Fw = n_ctx + 1
    act = int(cfg.dataset.spec.action_dim) * int(cfg.dataset.frame_interval)
    steps = int(cfg.experiment.diffusion.diffusion_steps)
    t_stab = int(round(0.02 * (steps - 1)))

    print(f"\n{'='*78}\n{domain}  arch={cfg.model.arch}  full_F={F}  trim_F={Fw}  n_ctx={n_ctx}  "
          f"causal={causal}\n{'='*78}")

    torch.manual_seed(7)
    ctx = torch.randn(1, n_ctx, lc, ls, ls, device="cuda")
    tgtA = torch.randn(1, 1, lc, ls, ls, device="cuda")   # prime target
    tgtB = torch.randn(1, 1, lc, ls, ls, device="cuda")   # measured/used target
    rest = torch.randn(1, F - Fw, lc, ls, ls, device="cuda") if F > Fw else None
    af = torch.randn(1, F, act, device="cuda") if cfg.model.use_action else None
    a_trim = af[:, :Fw].contiguous() if af is not None else None

    def win(target):
        return torch.cat([ctx, target], dim=1)                       # (1, Fw, ...)
    x_trim = win(tgtB)
    x_full = torch.cat([x_trim, rest], dim=1) if rest is not None else x_trim

    tt_trim = torch.full((1, Fw), steps // 2, device="cuda", dtype=torch.long); tt_trim[:, :n_ctx] = t_stab
    if F > Fw:
        tt_rest = torch.randint(0, steps, (1, F - Fw), device="cuda")
        tt_full = torch.cat([tt_trim, tt_rest], dim=1)
    else:
        tt_full = tt_trim

    def tgt(o):
        return o[:, n_ctx:n_ctx + 1].float()

    # ---- exactness (fp32, no autocast) ----
    with torch.no_grad():
        model.a5_configure(0, enabled=False); model._a2 = False
        base_tgt = tgt(model(x_full, tt_full, action=af))                       # V0 upstream target
        e_a1 = (tgt(model(x_trim, tt_trim, action=a_trim)) - base_tgt).abs().max().item()
        model._a2 = True
        e_a2 = (tgt(model(x_trim, tt_trim, action=a_trim)) - base_tgt).abs().max().item()
        if causal:
            model.a5_configure(n_ctx, enabled=True)
            model(win(tgtA), tt_trim, action=a_trim)                            # prime
            a5_fp32_tgt = tgt(model(win(tgtB), tt_trim, action=a_trim))         # use
            e_a5 = (a5_fp32_tgt - base_tgt).abs().max().item()
        else:
            a5_fp32_tgt = tgt(model(x_trim, tt_trim, action=a_trim)); e_a5 = e_a2

    # bf16 full stack target vs fp32 full stack target (the only lossy delta)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        if causal:
            model.a5_configure(n_ctx, enabled=True)
            model(win(tgtA), tt_trim, action=a_trim)
            bf16_tgt = tgt(model(win(tgtB), tt_trim, action=a_trim))
        else:
            model._a2 = True; bf16_tgt = tgt(model(x_trim, tt_trim, action=a_trim))
    e_bf16_vs_fp32 = (bf16_tgt - a5_fp32_tgt).abs().max().item()
    e_bf16_vs_base = (bf16_tgt - base_tgt).abs().max().item()

    print(f"  exactness (target-frame max_abs):")
    print(f"    +A1 fp32 vs V0        = {e_a1:.2e}")
    print(f"    +A2 fp32 vs V0        = {e_a2:.2e}")
    print(f"    +A5 fp32 vs V0        = {e_a5:.2e}   (full exact stack vs upstream)")
    print(f"    bf16 stack vs fp32    = {e_bf16_vs_fp32:.2e}   (only lossy step)")
    print(f"    bf16 stack vs V0      = {e_bf16_vs_base:.2e}")

    # ---- latency waterfall (per-step DiT forward) ----
    def measure():
        model.a5_configure(0, enabled=False); model._a2 = False
        v0 = timed(lambda: model(x_full, tt_full, action=af))                   # fp32 full
        v1 = timed(lambda: model(x_trim, tt_trim, action=a_trim))               # fp32 trim (A1)
        model._a2 = True
        v2 = timed(lambda: model(x_trim, tt_trim, action=a_trim))               # fp32 trim +A2
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v3 = timed(lambda: model(x_trim, tt_trim, action=a_trim))           # bf16 trim +A2 (B1)
            if causal:
                model.a5_configure(n_ctx, enabled=True)
                model(win(tgtA), tt_trim, action=a_trim)                        # prime once
                v4_use = timed(lambda: model(win(tgtB), tt_trim, action=a_trim))# bf16 +A5 use
                model.a5_configure(n_ctx, enabled=True)
                v4_prime = timed(lambda: (model.a5_configure(n_ctx, enabled=True),
                                          model(win(tgtA), tt_trim, action=a_trim))[1])
            else:
                v4_use = v3; v4_prime = v3
        model.a5_configure(0, enabled=False)
        return v0, v1, v2, v3, v4_use, v4_prime

    v0, v1, v2, v3, v4u, v4p = measure()
    layers = [("V0 baseline  fp32 full   ", v0), ("+A1 trim     fp32       ", v1),
              ("+A2          fp32       ", v2), ("+B1 bf16                ", v3),
              ("+A5 bf16 use            ", v4u)]
    print(f"  per-step DiT forward (ms) | cumulative x (vs V0) | marginal x:")
    prev = v0
    for name, ms in layers:
        print(f"    {name} {ms:8.1f} ms   {v0/ms:6.2f}x   ({prev/ms:4.2f}x)")
        prev = ms
    if v4u != v3:
        print(f"    (A5 prime forward       {v4p:8.1f} ms, paid once per window)")

    # ---- per-generated-frame projection ----
    def per_frame(use_ms, prime_ms, S, a5):
        return (prime_ms + (S - 1) * use_ms) if a5 else (S * use_ms)
    base_pf = per_frame(v0, v0, ship_steps, a5=False)
    stack_pf_ship = per_frame(v4u, v4p, ship_steps, a5=causal)
    stack_pf_c1 = per_frame(v4u, v4p, c1_steps, a5=causal)
    print(f"  per-generated-frame (sampling loop):")
    print(f"    V0 baseline  @ {ship_steps} steps        = {base_pf/1000:7.3f} s   1.00x")
    print(f"    full stack   @ {ship_steps} steps        = {stack_pf_ship/1000:7.3f} s   "
          f"{base_pf/stack_pf_ship:6.2f}x   (stack only, same steps)")
    print(f"    full stack + C1 @ {c1_steps} steps     = {stack_pf_c1/1000:7.3f} s   "
          f"{base_pf/stack_pf_c1:6.2f}x   (stack + fewer steps)")

    print("JSON " + json.dumps({
        "domain": domain, "arch": str(cfg.model.arch), "full_F": F, "trim_F": Fw,
        "per_step_ms": {"v0_fp32_full": v0, "a1_fp32_trim": v1, "a2_fp32": v2,
                        "b1_bf16": v3, "a5_bf16_use": v4u, "a5_bf16_prime": v4p},
        "exact": {"a1_vs_v0": e_a1, "a2_vs_v0": e_a2, "a5_vs_v0": e_a5,
                  "bf16_vs_fp32": e_bf16_vs_fp32, "bf16_vs_v0": e_bf16_vs_base},
        "ship_steps": ship_steps, "c1_steps": c1_steps,
        "per_frame_s": {"baseline": base_pf / 1000, "stack_ship": stack_pf_ship / 1000,
                        "stack_c1": stack_pf_c1 / 1000},
        "speedup": {"stack_only": base_pf / stack_pf_ship, "stack_plus_c1": base_pf / stack_pf_c1},
    }))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="+", default=["dino_wm_pusht", "csgo"])
    ap.add_argument("--results_dir", default="/models/results")
    ap.add_argument("--opt", default="/repos/nanowm/src/models/nanowm.py")
    ap.add_argument("--ship_steps", type=int, default=50)
    # C1 recommended: control domains 16, csgo 25 (conservative). Per-domain override below.
    ap.add_argument("--c1_steps", type=int, default=None)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    for d in args.domains:
        c1 = args.c1_steps if args.c1_steps is not None else (25 if d == "csgo" else 16)
        run_domain(d, args.results_dir, args.opt, args.ship_steps, c1)


if __name__ == "__main__":
    raise SystemExit(main())
