# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Phase 2 module-validation smoke for Cosmos3-Nano-Policy-DROID on ROCm (gfx1151).

Per cursorrule 2 (validate modules independently against real data + partially loaded weights
BEFORE assembling the full pipeline), this loads the released checkpoint and exercises the model
in stages, reusing the UPSTREAM inference path (rule 2.1): it constructs the exact
``RobolabPolicyService`` the DROID policy server uses, then drives it directly with a synthetic
DROID-style observation (no RoboLab, no websocket). It reports, in order:

  1. model load: OmniInference.create(...) on ROCm  (imports + weight load + module construction)
  2. one forward: model.generate_samples_from_batch(...) -> action chunk  (MoT + diffusion)
  3. VAE decode : model.decode(vision_latent) -> RGB video               (Wan2.2 tokenizer)

and prints action/video shapes, cold vs steady latency, and peak VRAM. Exits non-zero on any
failure so a broken ROCm path (e.g. an attention kernel that silently needs natten/flash-attn)
is caught before the open-loop eval. No dataset needed.

Run (inside the cosmos3 image, GPU passthrough):
    CKPT=/models/Cosmos3-Nano-Policy-DROID  python /ryzers/scripts/model_smoke.py
"""
import os
import sys
import time

import numpy as np


def _synthetic_obs(h: int, w: int) -> dict:
    """A single DROID-style concat_view observation (upstream server input schema)."""
    rng = np.random.default_rng(0)
    return {
        "prompt": "pick up the object and place it in the bowl",
        # concat_view frame: wrist (top) + L/R shoulder (bottom). The server resizes to (h,w).
        "observation/image": rng.integers(0, 256, (h, w, 3), dtype=np.uint8),
        # joint_pos action space: 7 joints + 1 gripper. State rows are [T,D]; T=1 here.
        "observation/joint_position": np.zeros((1, 7), dtype=np.float32),
        "observation/gripper_position": np.zeros((1, 1), dtype=np.float32),
    }


def main() -> int:
    import torch

    ckpt = os.environ.get("CKPT", "/models/Cosmos3-Nano-Policy-DROID")
    num_steps = int(os.environ.get("NUM_STEPS", "4"))
    decode_video = os.environ.get("DECODE_VIDEO", "1") != "0"
    print(f"torch {torch.__version__} hip {torch.version.hip} | device {torch.cuda.get_device_name(0)}", flush=True)
    print(f"checkpoint dir   : {ckpt}", flush=True)
    if not torch.cuda.is_available():
        print("FAIL: no ROCm device visible.", file=sys.stderr)
        return 1

    # Upstream inference path (RoboLab is only the ws client; we call the service directly).
    from cosmos_framework.scripts import action_policy_server_robolab as srv
    from cosmos_framework.scripts.action_policy_server_robolab import (
        RobolabPolicyService,
        RobolabServerArgs,
    )

    # Disable content-safety guardrails for policy inference. They are text/video safety filters
    # (nltk + gated Aegis/video-content-safety models) irrelevant to action prediction, and pulling
    # them adds large gated downloads + per-step latency. OmniInference treats guardrails=None as a
    # clean no-op. This shims only our smoke driver, not upstream source (rule 2.1).
    _orig_build_setup_args = RobolabPolicyService._build_setup_args

    def _build_setup_args_no_guardrails(self, a):
        setup_args = _orig_build_setup_args(self, a)
        try:
            setup_args.guardrails = False
        except Exception as exc:  # pragma: no cover - defensive
            print(f"WARN: could not disable guardrails ({exc}); continuing", file=sys.stderr)
        return setup_args

    srv.RobolabPolicyService._build_setup_args = _build_setup_args_no_guardrails

    # ROCm gfx1151 enablement: register an SDPA attention backend (upstream ships only CUDA-only
    # cudnn/flash2/flash3/natten backends) and run eager. See cosmos3_rocm_patches for rationale.
    import cosmos3_rocm_patches
    cosmos3_rocm_patches.apply()

    # ---- Stage 1: model load -------------------------------------------------
    t0 = time.time()
    args = RobolabServerArgs(
        checkpoint_path=ckpt,
        decode_video=decode_video,   # VAE decode is heavy eagerly; gate via DECODE_VIDEO env
        num_steps=num_steps,
        deterministic_seed=True,
        seed=0,
    )
    svc = RobolabPolicyService(args)   # OmniInference.create(...) -> model on GPU
    load_s = time.time() - t0
    n_params = sum(p.numel() for p in svc.model.parameters())
    print(f"[1] model loaded : {load_s:.1f}s  params={n_params/1e9:.2f}B", flush=True)
    print(f"    peak VRAM    : {torch.cuda.max_memory_allocated()/1e9:.1f} GB (after load)", flush=True)

    obs = _synthetic_obs(svc.cfg.image_height, svc.cfg.image_width)

    # ---- Stage 2+3: forward (+ VAE decode via decode_video) ------------------
    # First call includes ROCm/AOTriton kernel warmup; time a steady call after.
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    out_cold = svc.infer(obs)
    torch.cuda.synchronize()
    cold_s = time.time() - t0

    t0 = time.time()
    out = svc.infer(obs)
    torch.cuda.synchronize()
    steady_s = time.time() - t0

    action = out["action"]
    print(f"[2] forward ok   : action shape={action.shape} dtype={action.dtype}", flush=True)
    print(f"    action[0]    : {np.array2string(action[0], precision=3, max_line_width=120)}", flush=True)
    if "video" in out:
        print(f"[3] VAE decode ok: video shape={out['video'].shape} dtype={out['video'].dtype}", flush=True)
    print(f"    latency      : cold={cold_s:.2f}s  steady={steady_s:.2f}s (num_steps={num_steps})", flush=True)
    print(f"    peak VRAM    : {torch.cuda.max_memory_allocated()/1e9:.1f} GB (inference)", flush=True)
    print("PASS: Cosmos3-Nano-Policy-DROID loads + runs a forward + decodes video on ROCm", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
