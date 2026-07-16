# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Per-component runtime latency profile for Cosmos3-Nano-Policy-DROID on ROCm (gfx1151).

Loads the released policy via the upstream RobolabPolicyService (rule 2.1), warms up, then
profiles ONE steady action inference with CUDA-event forward hooks on every nn.Module to
attribute *inclusive* GPU time. Aggregates into a per-top-level-component table (VAE encode,
text/prompt path, MoT/DiT diffusion transformer x num_steps, action head, ...) plus the
top self-time hotspots, and dumps a JSON that the arch-diagram renderer consumes.

VAE decode is NOT run here: on gfx1151 / ROCm 7.2.2 the Wan2.2 3D-conv decode hangs in
bf16/fp16 (see artifacts/cosmos3_decode_bottleneck/RESULTS.md); its cost is annotated from
that analysis in the diagram.

Run inside the cosmos3 image (GPU passthrough), PYTHONPATH must see cosmos3_rocm_patches:
    CKPT=/models/Cosmos3-Nano-Policy-DROID NUM_STEPS=4 OUT_DIR=/outputs \
      python /work/cosmos3_latency_profile.py
"""
import collections
import json
import os
import sys
import time

import numpy as np


def _synthetic_obs(h: int, w: int) -> dict:
    rng = np.random.default_rng(0)
    return {
        "prompt": "pick up the object and place it in the bowl",
        "observation/image": rng.integers(0, 256, (h, w, 3), dtype=np.uint8),
        "observation/joint_position": np.zeros((1, 7), dtype=np.float32),
        "observation/gripper_position": np.zeros((1, 1), dtype=np.float32),
    }


def _nparams(mod) -> int:
    return sum(p.numel() for p in mod.parameters(recurse=True))


def _dump_tree(model, max_depth: int = 2):
    """Structured (name, class, params, direct-children) for the top levels -> for the diagram."""
    tree = []
    for name, child in model.named_children():
        tree.append({
            "name": name,
            "class": type(child).__name__,
            "params": _nparams(child),
            "children": [
                {"name": n2, "class": type(c2).__name__, "params": _nparams(c2)}
                for n2, c2 in child.named_children()
            ] if max_depth >= 2 else [],
        })
    return tree


def main() -> int:
    import torch

    ckpt = os.environ.get("CKPT", "/models/Cosmos3-Nano-Policy-DROID")
    num_steps = int(os.environ.get("NUM_STEPS", "4"))
    out_dir = os.environ.get("OUT_DIR", "/outputs")
    os.makedirs(out_dir, exist_ok=True)

    print(f"torch {torch.__version__} hip {torch.version.hip} | {torch.cuda.get_device_name(0)}", flush=True)
    print(f"checkpoint : {ckpt}  num_steps={num_steps}", flush=True)
    if not torch.cuda.is_available():
        print("FAIL: no ROCm device visible.", file=sys.stderr)
        return 1

    from cosmos_framework.scripts import action_policy_server_robolab as srv
    from cosmos_framework.scripts.action_policy_server_robolab import RobolabPolicyService, RobolabServerArgs

    _orig = RobolabPolicyService._build_setup_args

    def _no_guardrails(self, a):
        s = _orig(self, a)
        try:
            s.guardrails = False
        except Exception:
            pass
        return s

    srv.RobolabPolicyService._build_setup_args = _no_guardrails

    import cosmos3_rocm_patches
    cosmos3_rocm_patches.apply()

    # ---- load -----------------------------------------------------------------
    t0 = time.time()
    svc = RobolabPolicyService(
        RobolabServerArgs(
            checkpoint_path=ckpt, decode_video=False, num_steps=num_steps,
            deterministic_seed=True, seed=0,
        )
    )
    load_s = time.time() - t0
    model = svc.model
    total_params = _nparams(model)
    print(f"[load] {load_s:.1f}s  total_params={total_params/1e9:.2f}B", flush=True)
    print(f"[load] peak VRAM {torch.cuda.max_memory_allocated()/1e9:.1f} GB", flush=True)

    obs = _synthetic_obs(svc.cfg.image_height, svc.cfg.image_width)

    # ---- warmup (kernel autotune) + clean steady total ------------------------
    _ = svc.infer(obs)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    _ = svc.infer(obs)
    torch.cuda.synchronize()
    clean_total_s = time.time() - t0
    infer_vram_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"[steady] clean action-path latency = {clean_total_s:.2f}s  (num_steps={num_steps})", flush=True)

    # ---- hooked profile pass --------------------------------------------------
    name_of = {id(m): (n or "<root>") for n, m in model.named_modules()}
    params_of = {id(m): _nparams(m) for _, m in model.named_modules()}
    class_of = {id(m): type(m).__name__ for _, m in model.named_modules()}
    pending = collections.defaultdict(list)  # id(mod) -> [start_event, ...]
    pairs = []  # (id(mod), start_event, end_event)
    handles = []

    def pre_hook(mod, _inp):
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        pending[id(mod)].append(ev)

    def post_hook(mod, _inp, _out):
        st = pending[id(mod)]
        if not st:
            return
        start = st.pop()
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        pairs.append((id(mod), start, end))

    for _, m in model.named_modules():
        handles.append(m.register_forward_pre_hook(pre_hook))
        handles.append(m.register_forward_hook(post_hook))

    t0 = time.time()
    _ = svc.infer(obs)
    torch.cuda.synchronize()
    hooked_total_s = time.time() - t0
    for h in handles:
        h.remove()

    incl_ms = collections.defaultdict(float)
    calls = collections.defaultdict(int)
    for mid, start, end in pairs:
        incl_ms[mid] += start.elapsed_time(end)
        calls[mid] += 1

    # ---- per top-level component (inclusive) ----------------------------------
    comp = []
    for name, child in model.named_children():
        cid = id(child)
        comp.append({
            "name": name,
            "class": class_of.get(cid, type(child).__name__),
            "params": params_of.get(cid, _nparams(child)),
            "calls": calls.get(cid, 0),
            "incl_ms": incl_ms.get(cid, 0.0),
        })
    comp.sort(key=lambda c: c["incl_ms"], reverse=True)

    # ---- self-time hotspots (inclusive - sum(direct-children inclusive)) ------
    children_ids = collections.defaultdict(list)
    for n, m in model.named_modules():
        if not n:
            continue
        parent = n.rsplit(".", 1)[0] if "." in n else "<root>"
        children_ids[parent].append(id(m))
    self_ms = {}
    for n, m in model.named_modules():
        mid = id(m)
        kids = children_ids.get(n, [])
        self_ms[mid] = incl_ms.get(mid, 0.0) - sum(incl_ms.get(k, 0.0) for k in kids)
    hot = sorted(
        ((name_of[mid], class_of[mid], self_ms[mid], calls.get(mid, 0)) for mid in self_ms if self_ms[mid] > 1.0),
        key=lambda r: r[2], reverse=True,
    )[:25]

    # ---- report ---------------------------------------------------------------
    print("\n=== TOP-LEVEL COMPONENT LATENCY (inclusive GPU time, hooked pass) ===", flush=True)
    print(f"{'component':<28}{'class':<26}{'params':>10}{'calls':>7}{'incl_s':>10}{'%wall':>8}", flush=True)
    for c in comp:
        pct = 100.0 * c["incl_ms"] / 1000.0 / max(hooked_total_s, 1e-6)
        print(f"{c['name']:<28}{c['class']:<26}{c['params']/1e6:>9.1f}M{c['calls']:>7}"
              f"{c['incl_ms']/1000.0:>10.2f}{pct:>7.1f}%", flush=True)

    print("\n=== TOP SELF-TIME HOTSPOTS (leaf/near-leaf GPU cost) ===", flush=True)
    for nm, cl, sm, cc in hot:
        print(f"  {sm/1000.0:>7.2f}s  x{cc:<4} {cl:<24} {nm}", flush=True)

    print(f"\nclean action-path total : {clean_total_s:.2f}s", flush=True)
    print(f"hooked action-path total: {hooked_total_s:.2f}s (hook overhead {hooked_total_s-clean_total_s:+.2f}s)", flush=True)

    out = {
        "checkpoint": ckpt,
        "device": torch.cuda.get_device_name(0),
        "num_steps": num_steps,
        "total_params": int(total_params),
        "load_s": load_s,
        "clean_action_total_s": clean_total_s,
        "hooked_action_total_s": hooked_total_s,
        "infer_peak_vram_gb": infer_vram_gb,
        "components": comp,
        "hotspots": [
            {"name": nm, "class": cl, "self_s": sm / 1000.0, "calls": cc} for nm, cl, sm, cc in hot
        ],
        "module_tree": _dump_tree(model, max_depth=2),
        "cfg": {
            "action_dim": int(svc.cfg.action_dim),
            "action_chunk_size": int(getattr(svc.cfg, "action_chunk_size", 0)),
            "image_height": int(svc.cfg.image_height),
            "image_width": int(svc.cfg.image_width),
            "guidance": float(getattr(svc.cfg, "guidance", 0.0)),
            "shift": float(getattr(svc.cfg, "shift", 0.0)),
        },
    }
    mpath = os.path.join(out_dir, "cosmos3_latency_profile.json")
    with open(mpath, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {mpath}", flush=True)
    print("PROFILE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
