# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Full closed-loop step profile: WAN dream (compute_plan) vs VGGT Jacobian IDM + control.

profile_forward_mimicgen.py showed the warm WAN dream (generate_policy_chunk incl. cotracker) is
only ~20 s, yet a warm closed-loop chunk is ~142 s -> the remaining ~120 s must be the VGGT-1B
Jacobian IDM (+ adaptive control). This harness confirms that split: it builds the real
MotionPolicyGripper (start_server_mimicgen.build_policy), synthesizes one observation, and times
policy.predict_action_chunk() -- the exact two-stage call the server makes -- attributing:

  * dream  = compute_plan() wall (monkeypatched timer)
  * IDM+ctrl = predict_action_chunk wall - dream
  * VGGT forward-call count (hook) -> reveals Jacobian passes/step
  * op table (self CUDA) so the IDM's hot ops (addmm / attention / conv) are visible

Env: same as profile_forward_mimicgen.py (ALGO_CONFIG, SAMPLE_STEPS, VERA_* ckpt vars, OUT_DIR).
"""
import collections
import contextlib
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
SKIP_PROFILER = os.environ.get("SKIP_PROFILER", "").strip() in ("1", "true", "yes")
VIEW_KEYS = ["agentview_image", "robot0_eye_in_hand_image"]
H, VIEW_W = 128, 128


def main() -> int:
    print(f"torch {torch.__version__} hip {torch.version.hip} | {torch.cuda.get_device_name(0)}", flush=True)
    dev = torch.device("cuda:0")
    from vera.server.start_server_mimicgen import build_policy

    t0 = time.time()
    policy = build_policy(device=dev, algo_config_path=ALGO_CONFIG,
                          sample_steps_override=SAMPLE_STEPS, tracker_backend="cotracker")
    print(f"[build] {time.time()-t0:.1f}s ({type(policy).__name__})", flush=True)

    ctx_len = int(policy.motion_planner.required_pixel_frames)
    n_views = len(VIEW_KEYS)
    W = VIEW_W * n_views

    def make_obs():
        from vera.policy.base_policy import PolicyObservation
        context_rgb = (np.random.rand(ctx_len, H, W, 3).astype(np.float32))
        obs = PolicyObservation(
            rgb=context_rgb[-1][None].copy(),
            q_robot=np.zeros((1, 7), np.float32),
            view_keys=VIEW_KEYS, view_widths=[VIEW_W] * n_views,
            eef_pos=np.array([[0.0, 0.0, 1.0]], np.float32),
            eef_quat=np.array([[0.0, 0.0, 0.0, 1.0]], np.float32),
            gripper_qpos=np.zeros((1, 2), np.float32),
            dt=0.05, action_mode="velocity", step_index=5,
        )
        return obs, context_rgb

    # --- timers/counters ---
    dream_times = []
    cls = type(policy)
    # walk the MRO to find the class that defines compute_plan
    import types
    orig_compute_plan = cls.compute_plan

    def timed_compute_plan(self, obs):
        torch.cuda.synchronize(); t = time.time()
        r = orig_compute_plan(self, obs)
        torch.cuda.synchronize(); dream_times.append(time.time() - t)
        return r
    cls.compute_plan = timed_compute_plan

    # VGGT / IDM forward-call counter
    idm_calls = collections.Counter()
    handles = []
    dyn = getattr(policy, "dynamics_model", None)
    dyn_root = None
    for cand in (dyn, getattr(dyn, "model", None), getattr(dyn, "algo", None), getattr(dyn, "net", None)):
        if isinstance(cand, torch.nn.Module):
            dyn_root = cand; break
    if dyn_root is not None:
        for n, m in dyn_root.named_modules():
            if type(m).__name__ in ("VGGT", "Aggregator", "VGGTJacobian", "DPTHead", "ImageJacobian"):
                def _mk(nm):
                    def _c(*a, **k): idm_calls[nm] += 1
                    return _c
                handles.append(m.register_forward_hook(lambda mod, i, o, nm=n or type(m).__name__: idm_calls.update({nm: 1})))
        print(f"[idm] dyn_root={type(dyn_root).__name__} hooked "
              f"{len(handles)} VGGT-ish modules", flush=True)
    else:
        print(f"[idm] could not find dynamics nn.Module (dyn={type(dyn).__name__})", flush=True)

    def run_step():
        obs, ctx = make_obs()
        return policy.predict_action_chunk(obs, context_rgb=ctx, execute_horizon=10,
                                           context_update_mode="replace")

    # warmup (pays MIOpen/rocBLAS/aotriton autotune once)
    for i in range(N_WARMUP):
        tw = time.time()
        try:
            with torch.no_grad():
                _ = run_step()
            torch.cuda.synchronize()
            print(f"[warmup {i+1}] full step {time.time()-tw:.1f}s  (dream={dream_times[-1]:.1f}s "
                  f"idm+ctrl={time.time()-tw-dream_times[-1]:.1f}s)", flush=True)
        except Exception as ex:
            import traceback; traceback.print_exc()
            print(f"[warmup {i+1}] FAILED after {time.time()-tw:.1f}s: {ex}", flush=True)

    # measured (+ optional profiler)
    dream_times.clear()
    prof = None
    prof_ctx = contextlib.nullcontext()
    if not SKIP_PROFILER:
        from torch.profiler import profile, ProfilerActivity
        prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False)
        prof_ctx = prof

    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    ok = True
    with torch.no_grad(), prof_ctx:
        for _ in range(N_MEASURE):
            try:
                _ = run_step()
            except Exception as ex:
                ok = False
                import traceback; traceback.print_exc()
                print(f"[measure] step failed: {ex}", flush=True)
        torch.cuda.synchronize()
    wall = time.time() - t0
    for h in handles:
        h.remove()
    cls.compute_plan = orig_compute_plan
    peak = torch.cuda.max_memory_allocated() / 1e9

    dream_s = sum(dream_times)
    idm_ctrl_s = max(0.0, wall - dream_s)
    print(f"\n=== FULL STEP: {N_MEASURE} step(s) in {wall:.1f}s  peak {peak:.1f} GB (ok={ok}) ===", flush=True)
    print(f"  dream (compute_plan)   {dream_s:>7.1f}s  {100*dream_s/max(wall,1e-6):>5.1f}%", flush=True)
    print(f"  IDM + control (rest)   {idm_ctrl_s:>7.1f}s  {100*idm_ctrl_s/max(wall,1e-6):>5.1f}%", flush=True)
    print(f"  VGGT/IDM forward calls : {dict(idm_calls)} (total {sum(idm_calls.values())})", flush=True)

    report = {
        "device": torch.cuda.get_device_name(0), "sample_steps": SAMPLE_STEPS,
        "wall_s": wall, "dream_s": dream_s, "idm_ctrl_s": idm_ctrl_s, "peak_gb": peak,
        "idm_calls": dict(idm_calls), "ok": ok,
    }
    if prof is not None:
        print("\n=== TOP OPS BY SELF CUDA TIME (full step: dream + IDM) ===", flush=True)
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=30), flush=True)
        try:
            prof.export_chrome_trace(str(OUT_DIR / "vera_fullstep_trace.json"))
        except Exception:
            pass
    with open(OUT_DIR / "vera_fullstep_profile.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nPROFILE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
