# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Full-model smoke test for FastWAM on AMD Strix Halo (gfx1151).

Direct PyTorch port: builds the *real* FastWAM model (Wan2.2-TI2V-5B MoT: T5 text
encoder + Wan VAE + video/action DiT) via the upstream hydra config, loads the
released LIBERO checkpoint, and runs ONE `infer_action` on a synthetic observation
to prove the whole flow-matching action path executes end-to-end on ROCm.

This deliberately exercises the full model as a single unit (not submodules).
No simulator / dataset is required: the observation is synthetic, and we only
assert the predicted action chunk has the right shape and is finite.

First run downloads the Wan2.2 base weights (handled by the upstream loader into
DIFFSYNTH_MODEL_BASE_PATH) plus the released checkpoint. Env knobs:
  FASTWAM_REPO      = /repos/fastwam        (upstream source with configs/)
  CKPT              = /models/fastwam_release/libero_uncond_2cam224.pt
  CONFIG_NAME       = sim_libero            (hydra config with eval overrides)
  NUM_STEPS         = 20                    (flow-matching denoise steps; upstream
                                             infer_action default is 20)
We report both first-call (includes one-time ROCm kernel-JIT + attention autotune
warmup) and steady-state latency. Exits non-zero on any failure so `ryzers run` / CI
catches a broken image.
"""
import os
import sys
import time

import numpy as np
import torch

FASTWAM_REPO = os.environ.get("FASTWAM_REPO", "/repos/fastwam")
CKPT = os.environ.get("CKPT", "/models/fastwam_release/libero_uncond_2cam224.pt")
CONFIG_NAME = os.environ.get("CONFIG_NAME", "sim_libero")
NUM_STEPS = int(os.environ.get("NUM_STEPS") or "20")  # upstream infer_action default
PROMPT = os.environ.get("PROMPT", "pick up the object and place it")


def _compose_cfg():
    """Compose the upstream eval config outside hydra.main, registering the
    resolvers that eval_libero_single.py installs at import time."""
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

    cfg_dir = os.path.join(FASTWAM_REPO, "configs")
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

    if FASTWAM_REPO not in sys.path:
        sys.path.insert(0, FASTWAM_REPO)
    if not os.path.exists(CKPT):
        print(f"FAIL: checkpoint not found: {CKPT}\n"
              f"      run /ryzers/scripts/download_smoke.sh first.", file=sys.stderr)
        return 1

    from hydra.utils import instantiate

    cfg = _compose_cfg()

    # Dimensions come straight from the upstream config so the smoke tracks the
    # real LIBERO 2-cam setup (video_size 224x448, 33 frames, proprio 8, action 7).
    video_size = list(cfg.data.train.video_size)
    height, width = int(video_size[0]), int(video_size[1])
    num_frames = int(cfg.data.train.num_frames)
    action_horizon = num_frames - 1
    proprio_dim = int(cfg.data.train.processor.proprio_output_dim)
    action_dim = int(cfg.data.train.processor.action_output_dim)
    print(f"config           : {CONFIG_NAME}  HxW={height}x{width}  "
          f"action_horizon={action_horizon}  proprio_dim={proprio_dim}  action_dim={action_dim}")

    t0 = time.time()
    model = instantiate(cfg.model, model_dtype=torch.bfloat16, device="cuda")
    model.load_checkpoint(str(CKPT))
    model = model.to("cuda").eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model loaded     : {time.time() - t0:.1f}s  params={n_params / 1e9:.2f}B")

    # Synthetic single-frame observation in [-1, 1] + zero proprio.
    image = (torch.rand(1, 3, height, width) * 2.0 - 1.0)
    proprio = torch.zeros(1, proprio_dim)

    def _run():
        with torch.no_grad():
            return model.infer_action(
                prompt=PROMPT,
                input_image=image,
                action_horizon=action_horizon,
                proprio=proprio,
                num_inference_steps=NUM_STEPS,  # default forward settings
                seed=0,
                rand_device="cpu",
            )

    def _sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # First call pays one-time ROCm HIP kernel-JIT + attention autotune + allocator
    # warmup; time it separately from steady-state so the numbers are not conflated.
    _sync()
    t0 = time.time()
    _run()
    _sync()
    cold_ms = (time.time() - t0) * 1000.0

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    _sync()
    t1 = time.time()
    out = _run()
    _sync()
    warm_ms = (time.time() - t1) * 1000.0

    action = out["action"]
    action = action.detach().to(dtype=torch.float32, device="cpu").numpy()
    if action.ndim == 3 and action.shape[0] == 1:
        action = action[0]
    peak_gb = (torch.cuda.max_memory_allocated() / 1e9) if torch.cuda.is_available() else 0.0
    print(f"infer_action     : {action.shape}  (default steps={NUM_STEPS})")
    print(f"  first-call     : {cold_ms:.0f} ms  (incl. one-time ROCm warmup)")
    print(f"  steady-state   : {warm_ms:.0f} ms  peak={peak_gb:.1f} GB")

    if action.ndim != 2 or action.shape[-1] != action_dim:
        print(f"FAIL: expected (T, {action_dim}) action chunk, got {action.shape}", file=sys.stderr)
        return 1
    if not np.isfinite(action).all():
        print("FAIL: non-finite actions.", file=sys.stderr)
        return 1

    print("PASS: FastWAM full-model ROCm smoke OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
