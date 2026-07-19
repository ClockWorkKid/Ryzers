# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Post-optimization component profile of the MimicGen closed-loop step (gfx1151).

Unlike profile_fullstep_mimicgen.py (which profiles the STOCK path), this applies the two baked
gfx1151 speedups first -- conv override (cudnn off) + IDM bf16 autocast -- so the measured split
reflects the shipped ~15.5 s/step reality, and then attributes GPU time to the four real consumers
so we know where the *remaining* time goes before picking the next lever:

  * WAN DiT (WanModel) -- the dream denoise loop
  * WAN VAE (encode + decode) -- conv3d, on the critical path (tracker needs pixels)
  * cotracker -- 2-view motion tracker on the dreamed video
  * VGGT aggregator / DPT -- the Jacobian IDM (runs ~20x/step)

Method: CUDA-event inclusive timers via pre/post module hooks (the torch.profiler self-CUDA table
comes back all-zeros on this ROCm build, so we use events). Also splits wall into dream
(compute_plan) vs IDM+control. Env: ALGO_CONFIG, SAMPLE_STEPS, N_WARMUP/N_MEASURE, OUT_DIR,
VERA_DISABLE_CUDNN/VERA_IDM_BF16 (both default 1, matching the baked patch).
"""
import collections
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ALGO_CONFIG = os.environ.get("ALGO_CONFIG", "/models/vera-ckpts/mimicgen-wan-1.3b/algo_config.yaml")
OUT_DIR = Path(os.environ.get("OUT_DIR", "/outputs"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_STEPS = int(os.environ.get("SAMPLE_STEPS", "40"))
N_WARMUP = int(os.environ.get("N_WARMUP", "1"))
N_MEASURE = int(os.environ.get("N_MEASURE", "1"))
VIEW_KEYS = ["agentview_image", "robot0_eye_in_hand_image"]
H, VIEW_W = 128, 128

# component classname -> bucket label
BUCKETS = {
    "WanModel": "wan_dit", "WanModelDiT": "wan_dit", "DiT": "wan_dit",
    "WanVAE": "wan_vae", "Encoder3d": "wan_vae", "Decoder3d": "wan_vae",
    "AutoencoderKLWan": "wan_vae", "VAEModule": "wan_vae",
    "CoTrackerPredictor": "cotracker", "CoTrackerThreeOffline": "cotracker",
    "EvolvingCoTracker": "cotracker", "CoTracker": "cotracker",
    "Aggregator": "vggt_idm", "VGGT": "vggt_idm", "DPTHead": "vggt_dpt",
}


def _bucket_for(mod):
    return BUCKETS.get(type(mod).__name__)


def main() -> int:
    print(f"torch {torch.__version__} hip {torch.version.hip} | {torch.cuda.get_device_name(0)}", flush=True)
    dev = torch.device("cuda:0")

    # ---- opt 1: conv override (match baked default) ----
    if os.environ.get("VERA_DISABLE_CUDNN", "1") == "1":
        torch.backends.cudnn.enabled = False
        print("[opt] cudnn disabled (ATen conv im2col+GEMM)", flush=True)

    from vera.server.start_server_mimicgen import build_policy

    t0 = time.time()
    policy = build_policy(device=dev, algo_config_path=ALGO_CONFIG,
                          sample_steps_override=SAMPLE_STEPS, tracker_backend="cotracker")
    print(f"[build] {time.time()-t0:.1f}s ({type(policy).__name__})", flush=True)

    # ---- opt 2: IDM bf16 autocast (match baked default) ----
    if os.environ.get("VERA_IDM_BF16", "1") == "1":
        dyn = getattr(policy, "dynamics_model", None)
        model = getattr(dyn, "model", None)
        if isinstance(model, torch.nn.Module) and hasattr(model, "compute_jacobian"):
            _orig = model.compute_jacobian

            def _cj_bf16(input_obs, __orig=_orig):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = __orig(input_obs)
                if isinstance(out, torch.Tensor):
                    return out.float()
                if isinstance(out, (tuple, list)):
                    return type(out)(o.float() if isinstance(o, torch.Tensor) else o for o in out)
                return out
            model.compute_jacobian = _cj_bf16
            print("[opt] IDM compute_jacobian wrapped in bf16 autocast", flush=True)

    ctx_len = int(policy.motion_planner.required_pixel_frames)
    n_views = len(VIEW_KEYS)
    W = VIEW_W * n_views

    def make_obs():
        from vera.policy.base_policy import PolicyObservation
        context_rgb = np.random.rand(ctx_len, H, W, 3).astype(np.float32)
        obs = PolicyObservation(
            rgb=context_rgb[-1][None].copy(), q_robot=np.zeros((1, 7), np.float32),
            view_keys=VIEW_KEYS, view_widths=[VIEW_W] * n_views,
            eef_pos=np.array([[0.0, 0.0, 1.0]], np.float32),
            eef_quat=np.array([[0.0, 0.0, 0.0, 1.0]], np.float32),
            gripper_qpos=np.zeros((1, 2), np.float32),
            dt=0.05, action_mode="velocity", step_index=5,
        )
        return obs, context_rgb

    # ---- dream vs IDM split (wall) ----
    dream_times = []
    cls = type(policy)
    orig_compute_plan = cls.compute_plan

    def timed_compute_plan(self, obs):
        torch.cuda.synchronize(); t = time.time()
        r = orig_compute_plan(self, obs)
        torch.cuda.synchronize(); dream_times.append(time.time() - t)
        return r
    cls.compute_plan = timed_compute_plan

    call_counts = collections.Counter()
    ev_pairs = collections.defaultdict(list)   # bucket -> [(start_ev, end_ev)]
    ev_stack = collections.defaultdict(list)    # bucket -> [start_ev] (re-entrancy)
    handles = []
    found = collections.Counter()

    def run_step():
        obs, ctx = make_obs()
        return policy.predict_action_chunk(obs, context_rgb=ctx, execute_horizon=10,
                                           context_update_mode="replace")

    # ---- warmup FIRST (pays autotune AND builds the lazy cotracker _motion_tracker) ----
    for i in range(N_WARMUP):
        tw = time.time()
        try:
            with torch.no_grad():
                _ = run_step()
            torch.cuda.synchronize()
            d = dream_times[-1] if dream_times else -1.0
            print(f"[warmup {i+1}] full step {time.time()-tw:.1f}s (dream={d:.1f}s)", flush=True)
        except Exception as ex:
            import traceback; traceback.print_exc()
            print(f"[warmup {i+1}] FAILED: {ex}", flush=True)

    # ---- component CUDA-event inclusive timers (hooks set AFTER warmup so tracker exists) ----
    # roots: planner._model (DiT+VAE), dynamics_model.model (VGGT), planner._motion_tracker (cotracker)
    roots = []
    planner = getattr(policy, "motion_planner", None)
    dyn = getattr(policy, "dynamics_model", None)
    for obj in (planner, dyn):
        if obj is None:
            continue
        for sub in (obj, getattr(obj, "_model", None), getattr(obj, "model", None)):
            if isinstance(sub, torch.nn.Module):
                roots.append(sub)
    # tracker is lazily built; grab it (or force-build) and add every nn.Module under it
    trk = getattr(planner, "_motion_tracker", None)
    if trk is None and hasattr(planner, "_get_motion_tracker"):
        try:
            trk = planner._get_motion_tracker()
        except Exception as ex:
            print(f"[hooks] tracker build failed: {ex}", flush=True)
    if trk is not None:
        for tattr in vars(trk).values() if hasattr(trk, "__dict__") else []:
            if isinstance(tattr, torch.nn.Module):
                roots.append(("__tracker__", tattr))
    # normalize roots to (tag, module)
    norm_roots = [(r if isinstance(r, tuple) else (None, r)) for r in roots]

    seen = set()
    for tag, root in norm_roots:
        for n, m in root.named_modules():
            b = "cotracker" if tag == "__tracker__" else _bucket_for(m)
            # for the tracker subtree, only bucket the top module to avoid double counting
            if tag == "__tracker__" and m is not root:
                b = None
            if b is None or id(m) in seen:
                continue
            seen.add(id(m)); found[b] += 1

            def _pre(mod, _inp, _b=b):
                e = torch.cuda.Event(enable_timing=True); e.record(); ev_stack[_b].append(e)

            def _post(mod, _inp, _out, _b=b):
                st = ev_stack[_b]
                if not st:
                    return
                s = st.pop(); e = torch.cuda.Event(enable_timing=True); e.record()
                ev_pairs[_b].append((s, e)); call_counts[_b] += 1
            handles.append(m.register_forward_pre_hook(_pre))
            handles.append(m.register_forward_hook(_post))
    print(f"[hooks] buckets found: {dict(found)} | tracker={type(trk).__name__ if trk else None}", flush=True)

    # reset timers for measured pass
    dream_times.clear(); ev_pairs.clear(); ev_stack.clear(); call_counts.clear()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.time(); ok = True
    with torch.no_grad():
        for _ in range(N_MEASURE):
            try:
                _ = run_step()
            except Exception as ex:
                ok = False; import traceback; traceback.print_exc()
                print(f"[measure] step failed: {ex}", flush=True)
        torch.cuda.synchronize()
    wall = time.time() - t0

    incl_ms = collections.defaultdict(float)
    for b, pairs in ev_pairs.items():
        for s, e in pairs:
            incl_ms[b] += s.elapsed_time(e)
    for h in handles:
        h.remove()
    cls.compute_plan = orig_compute_plan
    peak = torch.cuda.max_memory_allocated() / 1e9

    dream_s = sum(dream_times)
    idm_ctrl_s = max(0.0, wall - dream_s)
    print(f"\n=== POST-OPT FULL STEP: {N_MEASURE} step(s) in {wall:.1f}s  peak {peak:.1f} GB (ok={ok}) ===", flush=True)
    print(f"  dream (compute_plan)   {dream_s:>7.1f}s  {100*dream_s/max(wall,1e-6):>5.1f}%", flush=True)
    print(f"  IDM + control (rest)   {idm_ctrl_s:>7.1f}s  {100*idm_ctrl_s/max(wall,1e-6):>5.1f}%", flush=True)
    print("\n  --- component inclusive GPU time (may overlap dream/rest split) ---", flush=True)
    for b, ms in sorted(incl_ms.items(), key=lambda kv: kv[1], reverse=True):
        print(f"    {ms/1000.0:>7.2f}s  {100*ms/1000.0/max(wall,1e-6):>5.1f}%  {b:<10} "
              f"(calls={call_counts[b]})", flush=True)

    report = {
        "device": torch.cuda.get_device_name(0), "sample_steps": SAMPLE_STEPS,
        "opts": {"cudnn_disabled": os.environ.get("VERA_DISABLE_CUDNN", "1") == "1",
                 "idm_bf16": os.environ.get("VERA_IDM_BF16", "1") == "1"},
        "wall_s": wall, "dream_s": dream_s, "idm_ctrl_s": idm_ctrl_s, "peak_gb": peak,
        "component_incl_s": {b: ms / 1000.0 for b, ms in incl_ms.items()},
        "component_calls": dict(call_counts), "buckets_found": dict(found), "ok": ok,
    }
    with open(OUT_DIR / "vera_postopt_profile.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {OUT_DIR / 'vera_postopt_profile.json'}\nPROFILE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
