# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Two-phase latency benchmark for AHA-WAM on Strix Halo (gfx1151).

AHA-WAM splits inference into an asynchronous slow planner and a fast executor:

  phase="video"  -> Observation-Guided Video-Context prefill (umt5-xxl text encode
                    + Wan VAE encode + video-DiT world prefill). This is the SLOW
                    branch; async serving runs it in the background off the image
                    stream so the control loop does not pay it per action.
  phase="action" -> one Action-DiT chunk (action_chunk_size control steps) that
                    reuses the prefilled video KV cache. This is the FAST branch on
                    the latency-critical path.

This measures steady-state (warm) latency of each branch and derives the executor
throughput (control Hz) = action_chunk_size * 1000 / action_chunk_latency_ms. It
also compares SDPA attention backends on the action path to show the
flash/AOTriton effect:
  - MATH   : force torch math kernel (no flash)
  - FLASH  : force flash/AOTriton (+ efficient/math fallback for masked ops)
  - default: torch auto-selects (prefers flash where eligible)

Env: AHAWAM_REPO, CKPT, CONFIG_NAME(sim_robotwin), NUM_STEPS(action denoise steps),
BENCH_ITERS(5), PROMPT, DTYPE(bf16). With the ODE-distilled AHA-WAM-Flash checkpoint
use NUM_STEPS=1.
"""
import os
import sys
import time
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

AHAWAM_REPO = os.environ.get("AHAWAM_REPO", "/repos/ahawam")
CKPT = os.environ.get("CKPT", "/models/ahawam_release/robotwin_ahawam.pt")
CONFIG_NAME = os.environ.get("CONFIG_NAME", "sim_robotwin")
NUM_STEPS = int(os.environ.get("NUM_STEPS") or "10")
ITERS = int(os.environ.get("BENCH_ITERS") or "5")
PROMPT = os.environ.get("PROMPT", "pick up the object and place it")
_DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32, "fp16": torch.float16}
DTYPE = _DTYPES[os.environ.get("DTYPE", "bf16").lower()]


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _timed(fn, n=ITERS, warmup=2):
    for _ in range(warmup):
        fn()
    _sync()
    t = time.time()
    for _ in range(n):
        fn()
    _sync()
    return (time.time() - t) / n * 1000.0


def _compose_cfg():
    from omegaconf import OmegaConf
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    for name, fn in (("eval", eval), ("max", lambda x: max(x)),
                     ("split", lambda s, idx: s.split("/")[int(idx)])):
        try:
            OmegaConf.register_new_resolver(name, fn, replace=True)
        except Exception:
            pass
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=os.path.join(AHAWAM_REPO, "configs"), version_base="1.3"):
        return compose(config_name=CONFIG_NAME, overrides=[f"ckpt={CKPT}"])


def main() -> int:
    print(f"torch {torch.__version__}  hip={torch.version.hip}  cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device: {torch.cuda.get_device_name(0)}")
    for n in ("flash_sdp_enabled", "mem_efficient_sdp_enabled", "math_sdp_enabled"):
        fn = getattr(torch.backends.cuda, n, None)
        print(f"  {n}: {fn() if fn else 'n/a'}")

    if AHAWAM_REPO not in sys.path:
        sys.path.insert(0, AHAWAM_REPO)
    if not os.path.exists(CKPT):
        print(f"FAIL: checkpoint not found: {CKPT}", file=sys.stderr)
        return 1

    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    cfg = _compose_cfg()
    height, width = (int(x) for x in cfg.data.train.video_size)
    action_horizon = int(cfg.model.action_horizon)
    proprio_dim = int(cfg.data.train.processor.proprio_output_dim)
    action_dim = int(cfg.data.train.processor.action_output_dim)

    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
    model_cfg.load_text_encoder = True

    t0 = time.time()
    model = instantiate(model_cfg, model_dtype=DTYPE, device="cuda")
    model.load_checkpoint(str(CKPT))
    model = model.to("cuda").eval()
    chunk_size = int(model.action_chunk_size)
    print(f"dtype={DTYPE}  model loaded: {time.time()-t0:.1f}s  "
          f"params={sum(p.numel() for p in model.parameters())/1e9:.2f}B")
    print(f"config={CONFIG_NAME}  HxW={height}x{width}  action_horizon={action_horizon}  "
          f"chunk_size={chunk_size}  action_dim={action_dim}  action_steps={NUM_STEPS}")
    if torch.cuda.is_available():
        print(f"weights VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB\n")

    image = (torch.rand(1, 3, height, width, device="cuda", dtype=model.torch_dtype) * 2.0 - 1.0)
    proprio = torch.zeros(1, proprio_dim, device="cuda", dtype=torch.float32)

    def _reset():
        if hasattr(model, "reset_history"):
            model.reset_history()
        if hasattr(model, "_inference_state"):
            model._inference_state = None

    def prefill_video():
        with torch.no_grad():
            model.infer_action(prompt=PROMPT, input_image=image, action_horizon=action_horizon,
                               negative_prompt="", text_cfg_scale=1.0, seed=0, rand_device="cpu",
                               tiled=False, phase="video", num_inference_steps=NUM_STEPS)

    def video_branch():
        # Full slow planner: text encode + VAE encode + video-DiT prefill.
        _reset()
        prefill_video()

    def action_chunk(steps):
        # Fast executor: one action chunk reusing the prefilled video KV cache.
        # Reset the chunk cursor so we repeatedly time chunk 0 against the same context.
        model._inference_state["next_chunk_index"] = 0
        with torch.no_grad():
            return model.infer_action(chunk_obs_image=image, chunk_proprio=proprio,
                                      sigma_shift=None, tiled=False, phase="action",
                                      num_inference_steps=steps)

    # Prime once (pays one-time ROCm HIP kernel-JIT + attention autotune) and keep the
    # video state resident for the action-branch timings.
    video_branch()
    action_chunk(NUM_STEPS)

    t_video = _timed(video_branch, n=max(2, ITERS // 2))
    # Re-prime video state (video_branch resets it during its own timing loop).
    video_branch()
    t_action_1 = _timed(lambda: action_chunk(1))
    t_action_n = _timed(lambda: action_chunk(NUM_STEPS)) if NUM_STEPS > 1 else t_action_1
    per_step = (t_action_n - t_action_1) / (NUM_STEPS - 1) if NUM_STEPS > 1 else t_action_1
    ctrl_hz = chunk_size * 1000.0 / t_action_n if t_action_n > 0 else 0.0

    print("== two-phase steady-state latency ==")
    print(f"  video prefill (slow planner, async-amortized) : {t_video:8.0f} ms")
    print(f"  action chunk  @ 1 step                         : {t_action_1:8.0f} ms")
    print(f"  action chunk  @ {NUM_STEPS} steps{'':<24}: {t_action_n:8.0f} ms")
    print(f"  action per-denoise-step                        : {per_step:8.1f} ms")
    print(f"  executor throughput ({chunk_size} ctrl steps/chunk)   : {ctrl_hz:8.1f} control Hz\n")

    backends = {
        "MATH":    sdpa_kernel([SDPBackend.MATH]),
        "FLASH":   sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]),
        "default": nullcontext(),
    }
    print(f"{'backend':>8} | {'action chunk @'+str(NUM_STEPS)+' steps':>22} | {'control Hz':>10}")
    print("-" * 50)
    results = {}
    for label, ctx in backends.items():
        try:
            video_branch()
            with ctx:
                action_chunk(NUM_STEPS)  # warm under this backend
                t = _timed(lambda: action_chunk(NUM_STEPS))
            results[label] = t
            print(f"{label:>8} | {t:19.0f} ms | {chunk_size*1000.0/t:10.1f}")
        except Exception as e:
            print(f"{label:>8} | FAILED -> {type(e).__name__}: {str(e)[:60]}")

    # Correctness sanity.
    video_branch()
    out = action_chunk(NUM_STEPS)
    a = out["action_chunk"].detach().float().cpu().numpy()
    a = a[0] if (a.ndim == 3 and a.shape[0] == 1) else a
    if a.ndim != 2 or a.shape[-1] != action_dim or not np.isfinite(a).all():
        print(f"FAIL: bad action chunk shape/finite: {a.shape}", file=sys.stderr)
        return 1

    if "default" in results and "MATH" in results and results["default"] > 0:
        print(f"\nflash/default vs math action-chunk speedup: {results['MATH']/results['default']:.2f}x")
    print("PASS: two-phase planning benchmark complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
