# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""A/B prototype: cache WAN cross-attn K/V across the denoise loop (playbook lever 2.1).

The WAN DiT is the dominant post-opt cost (~56% of the closed-loop step, vera_postopt_profile.json).
Each of the 40 denoise steps calls the model with the SAME conditioning `context`, yet
WanT2VCrossAttention.forward recomputes k(context)/v(context) every step. Those projections depend
ONLY on the (step-invariant) context, so they can be computed once per plan and reused -- bit-exact.

This harness:
  * builds the real MimicGen planner (with the two baked gfx1151 speedups, matching production),
  * times the WAN DiT (WanModel forward, CUDA-event inclusive) over one dream, captures the output,
  * installs the per-plan K/V cache on every WanT2VCrossAttention (+ WanI2VCrossAttention img K/V),
  * re-times + re-captures with the SAME seed,
  * reports DiT speedup and output fidelity (expect max|Δ| ~ 0 / cos ~ 1.0 -- bit-exact reuse).

Env: ALGO_CONFIG, SAMPLE_STEPS (40), N_WARMUP (1), N_MEASURE (2), OUT_DIR, VERA_DISABLE_CUDNN/
VERA_IDM_BF16 (default 1). Nothing is shipped here; this only measures the lever.
"""
import collections
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

ALGO_CONFIG = os.environ.get("ALGO_CONFIG", "/models/vera-ckpts/mimicgen-wan-1.3b/algo_config.yaml")
OUT_DIR = Path(os.environ.get("OUT_DIR", "/outputs"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_STEPS = int(os.environ.get("SAMPLE_STEPS", "40"))
N_WARMUP = int(os.environ.get("N_WARMUP", "1"))
N_MEASURE = int(os.environ.get("N_MEASURE", "2"))
VIEW_KEYS = ["agentview_image", "robot0_eye_in_hand_image"]
H, VIEW_W = 128, 128

# shared plan epoch: bumped at the start of every dream so each plan gets fresh K/V, reused within
_STATE = {"epoch": 0}


def _install_kv_cache(model):
    """Monkeypatch every WanT2VCrossAttention / WanI2VCrossAttention to cache k(context)/v(context)
    per plan epoch. Returns a restore() closure."""
    import vera.video_model.algorithms.wan.modules.model as M
    attention = M.attention
    originals = []
    n_patched = 0
    for mod in model.modules():
        cls = type(mod).__name__
        if cls not in ("WanT2VCrossAttention", "WanI2VCrossAttention"):
            continue
        mod._kv_cache = {}
        originals.append((mod, mod.forward))

        if cls == "WanT2VCrossAttention":
            def _fwd(x, context, context_lens, _m=mod):
                b, n, d = x.size(0), _m.num_heads, _m.head_dim
                c = _m._kv_cache
                if c.get("epoch") != _STATE["epoch"]:
                    c["k"] = _m.norm_k(_m.k(context)).view(b, -1, n, d)
                    c["v"] = _m.v(context).view(b, -1, n, d)
                    c["clen"] = context_lens
                    c["epoch"] = _STATE["epoch"]
                k, v, clen = c["k"], c["v"], c["clen"]
                q = _m.norm_q(_m.q(x)).view(b, -1, n, d)
                out = attention(q, k, v, k_lens=clen)
                return _m.o(out.flatten(2))
            mod.forward = _fwd
        else:  # WanI2VCrossAttention: split context, cache both text + img K/V
            def _fwd(x, context, context_lens, _m=mod):
                b, n, d = x.size(0), _m.num_heads, _m.head_dim
                c = _m._kv_cache
                if c.get("epoch") != _STATE["epoch"]:
                    ctx_img = context[:, :257]
                    ctx = context[:, 257:]
                    c["k"] = _m.norm_k(_m.k(ctx)).view(b, -1, n, d)
                    c["v"] = _m.v(ctx).view(b, -1, n, d)
                    c["ki"] = _m.norm_k_img(_m.k_img(ctx_img)).view(b, -1, n, d)
                    c["vi"] = _m.v_img(ctx_img).view(b, -1, n, d)
                    c["clen"] = context_lens
                    c["epoch"] = _STATE["epoch"]
                q = _m.norm_q(_m.q(x)).view(b, -1, n, d)
                img_x = attention(q, c["ki"], c["vi"], k_lens=None).flatten(2)
                out = attention(q, c["k"], c["v"], k_lens=c["clen"]).flatten(2)
                return _m.o(out + img_x)
            mod.forward = _fwd
        n_patched += 1

    def restore():
        for mod, fwd in originals:
            mod.forward = fwd
            if hasattr(mod, "_kv_cache"):
                del mod._kv_cache
    return n_patched, restore


def _first_tensor(obj):
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, dict):
        for v in obj.values():
            t = _first_tensor(v)
            if t is not None:
                return t
    if isinstance(obj, (list, tuple)):
        for v in obj:
            t = _first_tensor(v)
            if t is not None:
                return t
    return None


def main() -> int:
    print(f"torch {torch.__version__} hip {torch.version.hip} | {torch.cuda.get_device_name(0)}", flush=True)
    dev = torch.device("cuda:0")
    if os.environ.get("VERA_DISABLE_CUDNN", "1") == "1":
        torch.backends.cudnn.enabled = False
        print("[opt] cudnn disabled", flush=True)

    from vera.server.start_server_mimicgen import build_policy
    t0 = time.time()
    policy = build_policy(device=dev, algo_config_path=ALGO_CONFIG,
                          sample_steps_override=SAMPLE_STEPS, tracker_backend="cotracker")
    print(f"[build] {time.time()-t0:.1f}s", flush=True)

    planner = policy.motion_planner
    model_root = getattr(planner, "_model", planner)
    # find the WanModel DiT
    wan = None
    for n, m in model_root.named_modules():
        if type(m).__name__ in ("WanModel", "WanModelDiT"):
            wan = m; break
    print(f"[dit] WanModel found: {wan is not None}", flush=True)

    ctx_len = int(planner.required_pixel_frames)
    horizon = int(planner.future_pixel_frames)
    n_views = len(VIEW_KEYS)
    W = VIEW_W * n_views

    def make_ctx():
        return torch.rand(1, ctx_len, 3, H, W)

    # DiT inclusive timer via WanModel forward hook (CUDA events)
    ev = {"pairs": [], "stack": []}

    def _pre(m, i):
        e = torch.cuda.Event(enable_timing=True); e.record(); ev["stack"].append(e)

    def _post(m, i, o):
        if ev["stack"]:
            s = ev["stack"].pop(); e = torch.cuda.Event(enable_timing=True); e.record()
            ev["pairs"].append((s, e))

    def run_dream(seed):
        torch.manual_seed(seed)
        _STATE["epoch"] += 1  # fresh plan -> fresh K/V (only matters when cache installed)
        return planner.generate_policy_chunk(make_ctx().clamp(0, 1), horizon=horizon,
                                             view_keys=VIEW_KEYS, view_widths=[VIEW_W] * n_views,
                                             text=None)

    def timed_runs(label, n):
        ev["pairs"].clear(); ev["stack"].clear()
        h1 = wan.register_forward_pre_hook(_pre) if wan else None
        h2 = wan.register_forward_hook(_post) if wan else None
        torch.cuda.synchronize(); t0 = time.time()
        out = None
        with torch.no_grad():
            for i in range(n):
                out = run_dream(seed=1234)  # fixed seed -> comparable outputs
            torch.cuda.synchronize()
        wall = time.time() - t0
        dit_ms = sum(s.elapsed_time(e) for s, e in ev["pairs"])
        if h1: h1.remove()
        if h2: h2.remove()
        print(f"[{label}] {n} dream(s) wall={wall:.2f}s  DiT_incl={dit_ms/1000:.2f}s  "
              f"({len(ev['pairs'])} WanModel calls)", flush=True)
        return wall, dit_ms / 1000.0, out

    # warmup (autotune)
    with torch.no_grad():
        for _ in range(N_WARMUP):
            _ = run_dream(seed=1)
    torch.cuda.synchronize()

    # ---- baseline ----
    base_wall, base_dit, base_out = timed_runs("baseline", N_MEASURE)
    base_t = _first_tensor(base_out)

    # ---- install K/V cache + measure ----
    n_patched, restore = _install_kv_cache(model_root)
    print(f"[kv-cache] patched {n_patched} cross-attn modules", flush=True)
    kv_wall, kv_dit, kv_out = timed_runs("kv_cache", N_MEASURE)
    kv_t = _first_tensor(kv_out)

    # ---- fidelity ----
    fid = {}
    if base_t is not None and kv_t is not None and base_t.shape == kv_t.shape:
        a = base_t.float().flatten(); b = kv_t.float().flatten()
        fid["max_abs"] = (a - b).abs().max().item()
        fid["cos"] = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
        fid["shape"] = list(base_t.shape)
    else:
        fid["note"] = f"shape mismatch base={None if base_t is None else list(base_t.shape)} kv={None if kv_t is None else list(kv_t.shape)}"
    restore()

    dit_speedup = base_dit / max(kv_dit, 1e-6)
    print(f"\n=== CROSS-ATTN K/V CACHE A/B (steps={SAMPLE_STEPS}) ===", flush=True)
    print(f"  DiT incl:  {base_dit:.2f}s -> {kv_dit:.2f}s  ({dit_speedup:.3f}x)", flush=True)
    print(f"  dream wall:{base_wall:.2f}s -> {kv_wall:.2f}s", flush=True)
    print(f"  fidelity:  {fid}", flush=True)

    report = {
        "device": torch.cuda.get_device_name(0), "sample_steps": SAMPLE_STEPS,
        "n_measure": N_MEASURE, "n_patched": n_patched,
        "baseline_dit_s": base_dit, "kv_dit_s": kv_dit, "dit_speedup": dit_speedup,
        "baseline_wall_s": base_wall, "kv_wall_s": kv_wall, "fidelity": fid,
    }
    with open(OUT_DIR / "vera_crossattn_kv_ab.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {OUT_DIR / 'vera_crossattn_kv_ab.json'}\nAB_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
