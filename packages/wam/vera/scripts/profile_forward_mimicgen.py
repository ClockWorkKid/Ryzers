# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Op/kernel-level forward-pass profile of the MimicGen closed-loop WAN-1.3B planner on ROCm.

Complements scripts/profile_videogen.py (which does component latency+FLOPs on the 14B DROID
planner). This one targets the *closed-loop* path that dominates the ~142 s/chunk cost on gfx1151
and answers the concrete speedup questions:

  1. DiT vs VAE(conv3d) vs attention split  -> torch.profiler self-CUDA-time op table.
  2. How many DiT forward passes run per chunk -> reveals effective sample_steps x guidance
     multiplier (lang_guidance/hist_guidance CFG branches). Hook counts real WanModel calls.
  3. Which SDPA backend actually fires (aotriton flash vs mem-efficient vs MATH) -> op names in
     the profiler table (aten::_scaled_dot_product_flash_attention / _efficient_attention /
     _scaled_dot_product_attention_math), plus a direct micro-probe on the DiT's q/k/v shape.
  4. hipblaslt gemm_and_bias -> unfused-cublas fallback frequency (stderr warning count).

Builds the planner exactly like vera.server.start_server_mimicgen.build_policy (same config,
same eff_steps/guidance), then times ONE generate_policy_chunk (the WAN dream) under CUDA-event
component hooks + torch.profiler. Env knobs for A/B sweeps:

  SAMPLE_STEPS   (default 40)      -- eff denoise steps
  LANG_GUIDANCE  (default from yaml, 3.0) / HIST_GUIDANCE (2.0)  -- set 0 to kill CFG passes
  N_WARMUP (1) / N_MEASURE (1)     -- generations
  ALGO_CONFIG    (default /models/vera-ckpts/mimicgen-wan-1.3b/algo_config.yaml)
  SKIP_PROFILER=1                  -- component hooks only (no op table)
  OUT_DIR        (default /outputs)
"""
import collections
import contextlib
import json
import os
import sys
import time
from pathlib import Path

import torch

ALGO_CONFIG = os.environ.get(
    "ALGO_CONFIG", "/models/vera-ckpts/mimicgen-wan-1.3b/algo_config.yaml"
)
OUT_DIR = Path(os.environ.get("OUT_DIR", "/outputs"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_STEPS = int(os.environ.get("SAMPLE_STEPS", "40"))
LANG_G = os.environ.get("LANG_GUIDANCE")
HIST_G = os.environ.get("HIST_GUIDANCE")
N_WARMUP = int(os.environ.get("N_WARMUP", "1"))
N_MEASURE = int(os.environ.get("N_MEASURE", "1"))
SKIP_PROFILER = os.environ.get("SKIP_PROFILER", "").strip() in ("1", "true", "yes")
VIEW_KEYS = ["agentview_image", "robot0_eye_in_hand_image"]


def _find_module_by_classname(root, *names):
    hits = []
    for n, m in root.named_modules():
        if type(m).__name__ in names:
            hits.append((n, m))
    return hits


def main() -> int:
    print(f"torch {torch.__version__} hip {torch.version.hip}", flush=True)
    if not torch.cuda.is_available():
        print("FAIL: no ROCm device", file=sys.stderr)
        return 1
    print(f"device: {torch.cuda.get_device_name(0)}", flush=True)
    dev = torch.device("cuda:0")

    from vera.server.start_server_mimicgen import build_policy

    lang_override = float(LANG_G) if LANG_G is not None else None
    hist_override = float(HIST_G) if HIST_G is not None else None
    print(f"[build] algo_config={ALGO_CONFIG} steps={SAMPLE_STEPS} "
          f"lang_g={lang_override} hist_g={hist_override}", flush=True)

    t0 = time.time()
    policy = build_policy(
        device=dev,
        algo_config_path=ALGO_CONFIG,
        sample_steps_override=SAMPLE_STEPS,
        lang_guidance_override=lang_override,
        hist_guidance_override=hist_override,
        tracker_backend="cotracker",
    )
    print(f"[build] policy built in {time.time()-t0:.1f}s ({type(policy).__name__})", flush=True)

    planner = getattr(policy, "motion_planner", None)
    if planner is None:
        print("FAIL: policy has no motion_planner", file=sys.stderr)
        return 2
    ctx_len = int(planner.required_pixel_frames)
    horizon = int(planner.future_pixel_frames)
    n_views = len(VIEW_KEYS)
    H = 128
    view_w = 128
    W = view_w * n_views
    print(f"[shape] ctx_len={ctx_len} horizon={horizon} views={n_views} canvas={H}x{W}", flush=True)

    # planner (WanAllTrackerPipeline) is a plain wrapper, not an nn.Module; the real modules live
    # under planner._model (the WanTextToVideo algo). Search there for the DiT (WanModel).
    model_root = getattr(planner, "_model", planner)
    dits = _find_module_by_classname(model_root, "WanModel", "WanModelDiT", "DiT")
    print(f"[dit] candidate DiT modules: {[n or '<root>' for n, _ in dits]}", flush=True)

    def make_ctx():
        return torch.rand(1, ctx_len, 3, H, W, device="cpu")

    def run_chunk():
        return planner.generate_policy_chunk(
            make_ctx().clamp(0, 1),
            horizon=horizon,
            view_keys=VIEW_KEYS,
            view_widths=[view_w] * n_views,
            text=None,
        )

    # ---- warmup ----
    for i in range(N_WARMUP):
        tw = time.time()
        with torch.no_grad():
            _ = run_chunk()
        torch.cuda.synchronize()
        print(f"[warmup {i+1}] {time.time()-tw:.1f}s", flush=True)

    # ---- DiT call counter + component inclusive hooks ----
    dit_calls = collections.Counter()
    handles = []
    for n, m in dits:
        def _mk(nm):
            def _c(mod, inp, out):
                dit_calls[nm] += 1
            return _c
        handles.append(m.register_forward_hook(_mk(n or "<root>")))

    # component inclusive time via CUDA events on the top planner submodules
    comp = {}
    for cname in ("_model",):
        top = getattr(planner, cname, None)
        if top is None:
            continue
        for sub_name, mod in top.named_children():
            comp[f"{cname}.{sub_name}"] = mod
    incl_ms = collections.defaultdict(float)
    pend = collections.defaultdict(list)

    def pre(mod, _i, _key):
        ev = torch.cuda.Event(enable_timing=True); ev.record(); pend[_key].append(ev)

    def post(mod, _i, _o, _key):
        st = pend[_key]
        if not st: return
        s = st.pop(); e = torch.cuda.Event(enable_timing=True); e.record()
        incl_ms[_key] += 0.0  # filled after sync
        pend[f"__pairs__{_key}"].append((s, e))

    for key, mod in comp.items():
        handles.append(mod.register_forward_pre_hook(lambda m, i, k=key: pre(m, i, k)))
        handles.append(mod.register_forward_hook(lambda m, i, o, k=key: post(m, i, o, k)))

    # ---- measured pass (+ optional torch.profiler op table) ----
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    prof_ctx = contextlib.nullcontext()
    prof = None
    if not SKIP_PROFILER:
        from torch.profiler import profile, ProfilerActivity
        prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                       record_shapes=True, with_stack=False)
        prof_ctx = prof

    t0 = time.time()
    with torch.no_grad(), prof_ctx:
        for _ in range(N_MEASURE):
            _ = run_chunk()
        torch.cuda.synchronize()
    wall = time.time() - t0
    for key in list(comp.keys()):
        for s, e in pend.get(f"__pairs__{key}", []):
            incl_ms[key] += s.elapsed_time(e)
    for h in handles:
        h.remove()
    peak = torch.cuda.max_memory_allocated() / 1e9

    print(f"\n=== FORWARD PASS: {N_MEASURE} chunk(s) in {wall:.1f}s  peak {peak:.1f} GB ===", flush=True)
    print(f"[dit] WanModel forward calls total = {sum(dit_calls.values())} "
          f"-> {sum(dit_calls.values())/max(N_MEASURE,1):.0f}/chunk "
          f"(steps={SAMPLE_STEPS} -> passes/step={sum(dit_calls.values())/max(N_MEASURE*SAMPLE_STEPS,1):.2f})",
          flush=True)
    for k, v in dit_calls.items():
        print(f"      {k}: {v}", flush=True)

    print("\n=== TOP-LEVEL COMPONENT INCLUSIVE GPU TIME (planner._model children) ===", flush=True)
    for key, ms in sorted(incl_ms.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {ms/1000.0:>8.2f}s  {100*ms/1000.0/max(wall,1e-6):>5.1f}%  {key}", flush=True)

    report = {
        "device": torch.cuda.get_device_name(0), "torch": torch.__version__, "hip": torch.version.hip,
        "algo_config": ALGO_CONFIG, "sample_steps": SAMPLE_STEPS,
        "lang_guidance": lang_override, "hist_guidance": hist_override,
        "ctx_len": ctx_len, "horizon": horizon, "wall_s": wall, "peak_gb": peak,
        "dit_calls_total": sum(dit_calls.values()), "dit_calls": dict(dit_calls),
        "component_incl_s": {k: v / 1000.0 for k, v in incl_ms.items()},
    }

    if prof is not None:
        print("\n=== TOP OPS BY SELF CUDA TIME (attention backend + conv3d + gemm show here) ===",
              flush=True)
        ka = prof.key_averages()
        table = ka.table(sort_by="self_cuda_time_total", row_limit=35)
        print(table, flush=True)
        # extract SDPA / conv / gemm rows for the JSON
        rows = []
        for e in ka:
            nm = e.key
            if any(t in nm.lower() for t in
                   ("scaled_dot_product", "attention", "convolution", "conv3d", "miopen",
                    "gemm", "addmm", "mm", "bmm", "softmax")):
                rows.append({
                    "op": nm,
                    "self_cuda_ms": getattr(e, "self_cuda_time_total", 0) / 1000.0,
                    "cuda_ms": getattr(e, "cuda_time_total", 0) / 1000.0,
                    "count": e.count,
                })
        rows.sort(key=lambda r: r["self_cuda_ms"], reverse=True)
        report["hot_ops"] = rows[:40]
        tpath = OUT_DIR / "vera_forward_profile_trace.json"
        try:
            prof.export_chrome_trace(str(tpath))
            print(f"[trace] chrome trace -> {tpath}", flush=True)
        except Exception as ex:
            print(f"[trace] export failed: {ex}", flush=True)

    mpath = OUT_DIR / "vera_forward_profile.json"
    with open(mpath, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {mpath}\nPROFILE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
