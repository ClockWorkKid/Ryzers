# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Full-model smoke test for AHA-WAM on AMD Strix Halo (gfx1151).

Direct PyTorch port: builds the *real* AHA-WAM model (Wan2.2-TI2V-5B MoT: umt5-xxl
text encoder + Wan VAE + video-DiT world planner + action-DiT executor) via the
upstream hydra config, loads the released RoboTwin 2.0 checkpoint, and runs ONE
two-phase inference on a synthetic observation to prove the whole async
world-planner -> action-executor path executes end-to-end on ROCm:

  phase="video"  -> Observation-Guided Video-Context prefill (the slow planner);
  phase="action" -> one Action-DiT chunk conditioned on the reused context.

This deliberately exercises the full model as a single unit (not submodules).
No simulator / dataset is required: the observation is synthetic, and we only
assert the predicted action chunk has the right shape and is finite.

First run downloads the Wan2.2 base weights (handled by the upstream DiffSynth
loader into DIFFSYNTH_MODEL_BASE_PATH) plus the released checkpoint. Env knobs:
  AHAWAM_REPO  = /repos/ahawam        (upstream source with configs/)
  CKPT         = /models/ahawam_release/robotwin_ahawam.pt
  CONFIG_NAME  = sim_robotwin         (hydra config with eval overrides)
  NUM_STEPS    = 10                   (action denoise steps; upstream eval default)
  PROMPT       = "pick up the object and place it"
We report both first-call (includes one-time ROCm kernel-JIT + attention autotune
warmup) and steady-state latency. Exits non-zero on any failure so `ryzers run` / CI
catches a broken image.
"""
import os
import sys
import time

import numpy as np
import torch

AHAWAM_REPO = os.environ.get("AHAWAM_REPO", "/repos/ahawam")
CKPT = os.environ.get("CKPT", "/models/ahawam_release/robotwin_ahawam.pt")
CONFIG_NAME = os.environ.get("CONFIG_NAME", "sim_robotwin")
NUM_STEPS = int(os.environ.get("NUM_STEPS") or "10")  # upstream eval default
PROMPT = os.environ.get("PROMPT", "pick up the object and place it")


def _compose_cfg():
    """Compose the upstream eval config outside hydra.main."""
    from omegaconf import OmegaConf
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    for name, fn in (
        ("eval", eval),
        ("max", lambda x: max(x)),
        ("split", lambda s, idx: s.split("/")[int(idx)]),
    ):
        try:
            OmegaConf.register_new_resolver(name, fn, replace=True)
        except Exception:
            pass

    cfg_dir = os.path.join(AHAWAM_REPO, "configs")
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=cfg_dir, version_base="1.3"):
        cfg = compose(config_name=CONFIG_NAME, overrides=[f"ckpt={CKPT}"])
    return cfg


def main() -> int:
    print(f"torch            : {torch.__version__}  hip={torch.version.hip}")
    if not torch.version.hip:
        print("FAIL: torch is not a ROCm build.", file=sys.stderr)
        return 1
    if not torch.cuda.is_available():
        print("FAIL: no ROCm device visible (check /dev/kfd, /dev/dri).", file=sys.stderr)
        return 1
    print(f"device[0]        : {torch.cuda.get_device_name(0)}")

    if AHAWAM_REPO not in sys.path:
        sys.path.insert(0, AHAWAM_REPO)
    if not os.path.exists(CKPT):
        print(f"FAIL: checkpoint not found: {CKPT}\n"
              f"      run /ryzers/scripts/download_checkpoints.sh first.", file=sys.stderr)
        return 1

    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    cfg = _compose_cfg()

    # Dimensions come straight from the upstream config so the smoke tracks the real
    # RoboTwin 2.0 setup (video 384x320, 3-cam concat, proprio 14, action 14).
    video_size = list(cfg.data.train.video_size)
    height, width = int(video_size[0]), int(video_size[1])
    action_dim = int(cfg.data.train.processor.action_output_dim)
    proprio_dim = int(cfg.data.train.processor.proprio_output_dim)
    action_horizon = int(cfg.model.action_horizon)

    # Resolve the model subtree to a container so ${data...} interpolations bind, then
    # force the text encoder on for real inference (mirrors deploy/server/ahawam_policy).
    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
    model_cfg.load_text_encoder = True

    t0 = time.time()
    model = instantiate(model_cfg, model_dtype=torch.bfloat16, device="cuda")
    model.load_checkpoint(str(CKPT))
    model = model.to("cuda").eval()
    chunk_size = int(model.action_chunk_size)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model loaded     : {time.time() - t0:.1f}s  params={n_params / 1e9:.2f}B")
    print(f"config           : {CONFIG_NAME}  HxW={height}x{width}  action_horizon={action_horizon}  "
          f"chunk_size={chunk_size}  proprio_dim={proprio_dim}  action_dim={action_dim}")

    # Synthetic single front-cam observation in [-1, 1] + zero proprio.
    image = (torch.rand(1, 3, height, width, device="cuda", dtype=model.torch_dtype) * 2.0 - 1.0)
    proprio = torch.zeros(1, proprio_dim, device="cuda", dtype=torch.float32)

    def _reset():
        if hasattr(model, "reset_history"):
            model.reset_history()
        if hasattr(model, "_inference_state"):
            model._inference_state = None

    def _run():
        with torch.no_grad():
            # Slow planner: prefill the reusable video context (OVCR).
            model.infer_action(
                prompt=PROMPT,
                input_image=image,
                action_horizon=action_horizon,
                negative_prompt="",
                text_cfg_scale=1.0,
                seed=0,
                rand_device="cpu",
                tiled=False,
                phase="video",
                num_inference_steps=NUM_STEPS,
            )
            # Fast executor: one action chunk conditioned on the reused context.
            out = model.infer_action(
                chunk_obs_image=image,
                chunk_proprio=proprio,
                sigma_shift=None,
                tiled=False,
                phase="action",
                num_inference_steps=NUM_STEPS,
            )
        return out

    def _sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # First call pays one-time ROCm HIP kernel-JIT + attention autotune + allocator
    # warmup; time it separately from steady-state so the numbers are not conflated.
    _reset(); _sync()
    t0 = time.time(); _run(); _sync()
    cold_ms = (time.time() - t0) * 1000.0

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    _reset(); _sync()
    t1 = time.time(); out = _run(); _sync()
    warm_ms = (time.time() - t1) * 1000.0

    action = out["action_chunk"]
    action = action.detach().to(dtype=torch.float32, device="cpu").numpy()
    if action.ndim == 3 and action.shape[0] == 1:
        action = action[0]
    peak_gb = (torch.cuda.max_memory_allocated() / 1e9) if torch.cuda.is_available() else 0.0
    print(f"action_chunk     : {action.shape}  (action denoise steps={NUM_STEPS})")
    print(f"  first-call     : {cold_ms:.0f} ms  (incl. one-time ROCm warmup, video+action)")
    print(f"  steady-state   : {warm_ms:.0f} ms  peak={peak_gb:.1f} GB")

    if action.ndim != 2 or action.shape[-1] != action_dim:
        print(f"FAIL: expected (T, {action_dim}) action chunk, got {action.shape}", file=sys.stderr)
        return 1
    if not np.isfinite(action).all():
        print("FAIL: non-finite actions.", file=sys.stderr)
        return 1

    print("PASS: AHA-WAM full-model ROCm smoke OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
