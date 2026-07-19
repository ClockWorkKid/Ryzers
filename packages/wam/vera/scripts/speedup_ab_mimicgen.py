# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Kernel-level speedup A/B for the MimicGen closed-loop step on gfx1151 / ROCm.

Profiling established the warm closed-loop step (~150 s) is dominated by two ROCm-MIOpen/hipBLAS
pathologies, NOT algorithmic cost:
  * WAN VAE `conv3d` (bf16)  -> MIOpen has no tuned 3D solver -> NAIVE fallback kernel.
  * VGGT Jacobian IDM        -> loaded in **fp32** (build_policy never casts it) -> every gemm hits
    hipBLASLt's unsupported-config path -> unfused CUBLAS fallback + no flash-attention.

Two kernel-level levers (no change to sample_steps / horizon / any algorithm param):
  1. CONV_OVERRIDE : torch.backends.cudnn.enabled=False  -> forces PyTorch's ATen-native conv
     (im2col/unfold + GEMM via rocBLAS/hipBLASLt) instead of MIOpen's naive conv3d. This is the
     "basic pytorch override" that won our earlier cosmos3/strix conv3d microbenches.
  2. IDM bf16      : cast the VGGT IDM to bfloat16 (the WAN planner is already bf16). The Jacobian
     is a *learned dense field* emitted by ONE forward pass (vggt.aggregator + decoder) -- not a
     finite-difference Jacobian -- so bf16 carries no subtraction-cancellation risk. Kills the
     fp32 hipBLASLt fallback and lets attention hit the AOTriton flash kernel.

Single process, one policy build, an fp32 master of the IDM kept on CPU so every config is measured
from a clean fp32 baseline. For each config: warm once (pays autotune for that config's kernels),
then time N measured full steps (policy.predict_action_chunk -- the exact server call), splitting
dream (compute_plan) vs IDM+control. Also runs an IDM fp32-vs-bf16 fidelity check on a real
captured observation so we can confirm the Jacobian is preserved.

Env: CONFIGS (csv of baseline,conv,bf16,both,autocast,conv_autocast; default all), SAMPLE_STEPS(40),
N_WARMUP(1), N_MEASURE(2), ALGO_CONFIG, VERA_* ckpt vars, OUT_DIR. Sentinel: AB_DONE.
"""
import collections
import copy
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
N_MEASURE = int(os.environ.get("N_MEASURE", "2"))
VIEW_KEYS = ["agentview_image", "robot0_eye_in_hand_image"]
H, VIEW_W = 128, 128

# (name, cudnn_enabled, idm_mode)  idm_mode in {fp32, bf16, autocast}
ALL_CONFIGS = {
    "baseline":       (True,  "fp32"),      # stock: MIOpen conv + fp32 IDM
    "conv":           (False, "fp32"),      # + conv3d override only
    "bf16":           (True,  "bf16"),      # + IDM bf16 only
    "both":           (False, "bf16"),      # + conv override AND IDM bf16 (the stack)
    "autocast":       (True,  "autocast"),  # IDM under autocast (params fp32, ops bf16)
    "conv_autocast":  (False, "autocast"),
}
_sel = os.environ.get("CONFIGS", "baseline,conv,bf16,both").split(",")
CONFIGS = [(n, *ALL_CONFIGS[n]) for n in [s.strip() for s in _sel if s.strip()] if n in ALL_CONFIGS]


def main() -> int:
    print(f"torch {torch.__version__} hip {torch.version.hip} | {torch.cuda.get_device_name(0)}", flush=True)
    dev = torch.device("cuda:0")
    torch.manual_seed(0)
    from vera.server.start_server_mimicgen import build_policy

    t0 = time.time()
    policy = build_policy(device=dev, algo_config_path=ALGO_CONFIG,
                          sample_steps_override=SAMPLE_STEPS, tracker_backend="cotracker")
    print(f"[build] {time.time()-t0:.1f}s ({type(policy).__name__})", flush=True)

    ctx_len = int(policy.motion_planner.required_pixel_frames)
    n_views = len(VIEW_KEYS)
    W = VIEW_W * n_views

    # ---- locate the VGGT IDM nn.Module (policy.dynamics_model.model) + fp32 master ----
    dyn = getattr(policy, "dynamics_model", None)
    idm_model = getattr(dyn, "model", None)
    if not isinstance(idm_model, torch.nn.Module):
        idm_model = dyn if isinstance(dyn, torch.nn.Module) else None
    if idm_model is None:
        print("FAIL: could not find IDM nn.Module", file=sys.stderr); return 2
    master_sd = {k: v.detach().float().cpu().clone() for k, v in idm_model.state_dict().items()}
    orig_compute_jacobian = idm_model.compute_jacobian
    print(f"[idm] {type(idm_model).__name__}: {sum(p.numel() for p in idm_model.parameters())/1e6:.0f}M params, "
          f"master fp32 cached ({len(master_sd)} tensors)", flush=True)

    captured = {"obs": None}  # stash a real InputObservation for the fidelity check

    def set_idm_mode(mode):
        """Reload clean fp32 master, then apply the requested precision policy."""
        idm_model.float()
        idm_model.load_state_dict(master_sd)
        idm_model.to(dev)

        def _cast_rgb(input_obs, dt):
            try:
                input_obs.rgb = input_obs.rgb.to(dt)
            except Exception:
                object.__setattr__(input_obs, "rgb", input_obs.rgb.to(dt))
            return input_obs

        if mode == "fp32":
            def cj(input_obs):
                captured["obs"] = input_obs
                return orig_compute_jacobian(input_obs)
        elif mode == "bf16":
            idm_model.bfloat16()
            def cj(input_obs):
                captured["obs"] = input_obs
                return orig_compute_jacobian(_cast_rgb(input_obs, torch.bfloat16))
        elif mode == "autocast":
            def cj(input_obs):
                captured["obs"] = input_obs
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = orig_compute_jacobian(input_obs)
                # cast back to fp32 so the downstream fp32 control math doesn't hit a dtype
                # mismatch; the bf16 speedup already happened inside the autocast region.
                if isinstance(out, torch.Tensor):
                    return out.float()
                if isinstance(out, (tuple, list)):
                    return type(out)(o.float() if isinstance(o, torch.Tensor) else o for o in out)
                return out
        else:
            raise ValueError(mode)
        idm_model.compute_jacobian = cj

    # ---- dream timer (wraps compute_plan on the policy class) ----
    dream_times = []
    cls = type(policy)
    orig_compute_plan = cls.compute_plan

    def timed_compute_plan(self, obs):
        torch.cuda.synchronize(); t = time.time()
        r = orig_compute_plan(self, obs)
        torch.cuda.synchronize(); dream_times.append(time.time() - t)
        return r
    cls.compute_plan = timed_compute_plan

    def make_obs(seed):
        from vera.policy.base_policy import PolicyObservation
        rng = np.random.default_rng(seed)
        context_rgb = rng.random((ctx_len, H, W, 3), dtype=np.float32)
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

    def run_step(seed):
        obs, ctx = make_obs(seed)
        out = policy.predict_action_chunk(obs, context_rgb=ctx, execute_horizon=10,
                                          context_update_mode="replace")
        try:
            act = out["action"]
        except Exception:
            act = getattr(out, "action", None)
        a = None
        if act is not None:
            a = np.asarray(act, dtype=np.float32)
        return a

    results = []
    for name, cudnn_on, idm_mode in CONFIGS:
        torch.backends.cudnn.enabled = bool(cudnn_on)
        set_idm_mode(idm_mode)
        # warmup (autotune for this config)
        dream_times.clear()
        tw = time.time()
        try:
            with torch.no_grad():
                _ = run_step(1000)
            torch.cuda.synchronize()
            warm_s = time.time() - tw
        except Exception as ex:
            import traceback; traceback.print_exc()
            print(f"[{name}] WARMUP FAILED: {ex}", flush=True)
            results.append({"config": name, "cudnn": cudnn_on, "idm": idm_mode, "ok": False,
                            "error": str(ex)})
            continue

        # measured
        dream_times.clear()
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        act_last = None
        with torch.no_grad():
            for i in range(N_MEASURE):
                act_last = run_step(2000 + i)
            torch.cuda.synchronize()
        wall = (time.time() - t0) / max(N_MEASURE, 1)
        dream_s = sum(dream_times) / max(N_MEASURE, 1)
        idm_ctrl_s = max(0.0, wall - dream_s)
        peak = torch.cuda.max_memory_allocated() / 1e9
        act_norm = float(np.linalg.norm(act_last)) if act_last is not None else None
        act_finite = bool(np.all(np.isfinite(act_last))) if act_last is not None else None
        print(f"[{name:14s}] cudnn={cudnn_on!s:5s} idm={idm_mode:8s} | warm {warm_s:6.1f}s | "
              f"step {wall:7.2f}s (dream {dream_s:6.2f} idm+ctrl {idm_ctrl_s:7.2f}) | "
              f"peak {peak:.1f}GB | act_norm={act_norm} finite={act_finite}", flush=True)
        results.append({"config": name, "cudnn": cudnn_on, "idm": idm_mode, "ok": True,
                        "warm_s": warm_s, "step_s": wall, "dream_s": dream_s,
                        "idm_ctrl_s": idm_ctrl_s, "peak_gb": peak,
                        "act_norm": act_norm, "act_finite": act_finite})

    cls.compute_plan = orig_compute_plan

    # ---- IDM fidelity: fp32 vs bf16 vs autocast on the SAME captured observation ----
    fidelity = {}
    obs_ref = captured["obs"]
    if obs_ref is not None and hasattr(obs_ref, "rgb"):
        rgb_fp32 = obs_ref.rgb.detach().float().clone()

        def jac_for(mode):
            idm_model.float(); idm_model.load_state_dict(master_sd); idm_model.to(dev)
            o = copy.copy(obs_ref)
            if mode == "bf16":
                idm_model.bfloat16()
                try: o.rgb = rgb_fp32.to(torch.bfloat16)
                except Exception: object.__setattr__(o, "rgb", rgb_fp32.to(torch.bfloat16))
                with torch.no_grad():
                    return orig_compute_jacobian(o).detach().float()
            if mode == "autocast":
                try: o.rgb = rgb_fp32.clone()
                except Exception: object.__setattr__(o, "rgb", rgb_fp32.clone())
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    return orig_compute_jacobian(o).detach().float()
            try: o.rgb = rgb_fp32.clone()
            except Exception: object.__setattr__(o, "rgb", rgb_fp32.clone())
            with torch.no_grad():
                return orig_compute_jacobian(o).detach().float()

        try:
            j_ref = jac_for("fp32")
            denom = j_ref.abs().mean().clamp_min(1e-8)
        except Exception as ex:
            import traceback; traceback.print_exc()
            print(f"[fidelity] fp32 reference failed: {ex}", flush=True)
            j_ref = None
        if j_ref is not None:
            for m in ("autocast", "bf16"):  # per-mode guarded so one failure doesn't skip the rest
                try:
                    jm = jac_for(m)
                    rel = ((jm - j_ref).abs().mean() / denom).item()
                    mx = (jm - j_ref).abs().max().item()
                    cos = torch.nn.functional.cosine_similarity(
                        jm.flatten()[None], j_ref.flatten()[None]).item()
                    fidelity[m] = {"rel_mae": rel, "max_abs": mx, "cosine": cos}
                    print(f"[fidelity] IDM {m} vs fp32: rel_MAE={rel:.4e} max_abs={mx:.4e} cos={cos:.6f}", flush=True)
                except Exception as ex:
                    import traceback; traceback.print_exc()
                    print(f"[fidelity] {m} check failed: {ex}", flush=True)

    # ---- speedup table vs baseline ----
    base = next((r for r in results if r["config"] == "baseline" and r.get("ok")), None)
    print("\n=== SPEEDUP SUMMARY (step time, warm) ===", flush=True)
    for r in results:
        if not r.get("ok"):
            print(f"  {r['config']:14s}  FAILED", flush=True); continue
        sp = (base["step_s"] / r["step_s"]) if base else float("nan")
        print(f"  {r['config']:14s}  {r['step_s']:7.2f}s   {sp:5.2f}x   "
              f"(dream {r['dream_s']:.2f}s, idm+ctrl {r['idm_ctrl_s']:.2f}s)", flush=True)

    report = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__,
              "sample_steps": SAMPLE_STEPS, "n_measure": N_MEASURE,
              "configs": results, "fidelity": fidelity}
    with open(OUT_DIR / "vera_speedup_ab.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {OUT_DIR/'vera_speedup_ab.json'}\nAB_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
