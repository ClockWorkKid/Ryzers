# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Full-model smoke test for X-WAM on AMD Strix Halo (gfx1151).

Direct PyTorch port: builds the *real* X-WAM model (Wan2.2-TI2V-5B unified 4D WAM: UMT5-XXL
text encoder + Wan2.2 VAE + video/depth/action DiT via XWAMRunner), loads a released SFT
checkpoint, and runs ONE `generate(..., early_stop=True)` (the fast action-only path the
closed-loop controller uses) on a synthetic multi-view observation — proving the whole
asynchronous-denoising action path executes end-to-end on ROCm.

Mirrors evaluation/policy_server.py's load path (OmegaConf config from the exp dir, DeepSpeed
zero checkpoint under checkpoints/<steps>.ckpt/checkpoint/mp_rank_00_model_states.pt), but
uses a synthetic observation so no dataset is required. We assert the predicted action chunk
has the right shape and is finite, and report cold/steady latency + peak VRAM.

Requires weights (run scripts/download_checkpoints.sh first). Env knobs:
  XWAM_REPO   = /repos/xwam
  CKPT_ROOT   = /models/xwam/checkpoints          (has <exp>/config.yaml + checkpoints/)
  WAN_CKPT_DIR= /models/xwam/wan22_5b             (UMT5-XXL + Wan2.2 VAE + DiT base)
  EXP         = robotwin_sft                      (which SFT checkpoint to load)
  DENOISE_STEPS=50   ACTION_DENOISE_STEPS=10      (async denoising; the efficiency lever)
"""
import os
import sys
import time

import numpy as np
import torch

XWAM_REPO = os.environ.get("XWAM_REPO", "/repos/xwam")
CKPT_ROOT = os.environ.get("CKPT_ROOT", "/models/xwam/checkpoints")
WAN_CKPT_DIR = os.environ.get("WAN_CKPT_DIR", "/models/xwam/wan22_5b")
EXP = os.environ.get("EXP", "robotwin_sft")
STEPS = os.environ.get("STEPS", "last")
DENOISE_STEPS = int(os.environ.get("DENOISE_STEPS") or "50")
ACTION_DENOISE_STEPS = int(os.environ.get("ACTION_DENOISE_STEPS") or "10")
PROMPT = os.environ.get("PROMPT", "pick up the object and place it")
CFG = float(os.environ.get("CFG") or "0.0")


def main() -> int:
    print(f"torch            : {torch.__version__}  hip={torch.version.hip}")
    if not torch.version.hip:
        print("FAIL: torch is not a ROCm build.", file=sys.stderr)
        return 1
    if not torch.cuda.is_available():
        print("FAIL: no ROCm device visible (check /dev/kfd, /dev/dri).", file=sys.stderr)
        return 1
    print(f"device[0]        : {torch.cuda.get_device_name(0)}")

    if XWAM_REPO not in sys.path:
        sys.path.insert(0, XWAM_REPO)

    from omegaconf import OmegaConf
    import lightning as L
    from runners.xwam_runner import XWAMRunner

    exp_path = os.path.join(CKPT_ROOT, EXP)
    cfg_path = os.path.join(exp_path, "config.yaml")
    ckpt_path = os.path.join(exp_path, f"checkpoints/{STEPS}.ckpt/checkpoint/mp_rank_00_model_states.pt")
    for p in (cfg_path, ckpt_path):
        if not os.path.exists(p):
            print(f"FAIL: not found: {p}\n      run /ryzers/scripts/download_checkpoints.sh {EXP.replace('_sft','')} first.",
                  file=sys.stderr)
            return 1

    config = OmegaConf.load(cfg_path)
    config.sample_steps = DENOISE_STEPS
    config.use_decoupled_inference = ACTION_DENOISE_STEPS > 0
    config.action_denoise_steps = ACTION_DENOISE_STEPS
    config.action_num = config.dataset.frame_skip // config.dataset.action_skip
    config.wan_checkpoint_dir = WAN_CKPT_DIR

    video_size = list(config.dataset.video_size)  # [H, W]
    H, W = int(video_size[0]), int(video_size[1])
    num_views = 3
    proprio_dim = int(config.proprio_dim)
    print(f"config           : exp={EXP}  HxW={H}x{W}  views={num_views}  "
          f"proprio_dim={proprio_dim}  action_dim={config.action_dim}  action_num={config.action_num}")
    print(f"denoise          : video={DENOISE_STEPS}  action={ACTION_DENOISE_STEPS}  "
          f"decoupled={bool(config.use_decoupled_inference)}")

    L.seed_everything(int(config.seed), workers=True)

    t0 = time.time()
    model = XWAMRunner(config=config).cuda().bfloat16()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["module"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model loaded     : {time.time() - t0:.1f}s  params={n_params / 1e9:.2f}B")

    # Synthetic multi-view observation in [-1, 1] + zero (normalized) proprio.
    rgb = (torch.rand(1, num_views, 3, H, W) * 2.0 - 1.0).bfloat16().cuda()
    proprio = torch.zeros(1, proprio_dim).bfloat16().cuda()

    def _run():
        with torch.inference_mode():
            return model.generate(rgb, proprio, [PROMPT], seeds=[0], early_stop=True, cfg=CFG, run_depth=False)

    def _sync():
        torch.cuda.synchronize()

    _sync(); t0 = time.time(); _run(); _sync()
    cold_ms = (time.time() - t0) * 1000.0

    torch.cuda.reset_peak_memory_stats()
    _sync(); t1 = time.time(); out = _run(); _sync()
    warm_ms = (time.time() - t1) * 1000.0

    _, xt_actions, xt_proprios, _ = out
    action = xt_actions[0].detach().to(dtype=torch.float32, device="cpu").numpy()
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"generate(actions): {action.shape}")
    print(f"  first-call     : {cold_ms:.0f} ms  (incl. one-time ROCm warmup)")
    print(f"  steady-state   : {warm_ms:.0f} ms  peak={peak_gb:.1f} GB")

    if action.ndim != 2 or action.shape[-1] != int(config.action_dim):
        print(f"FAIL: expected (Ta, {int(config.action_dim)}) action chunk, got {action.shape}", file=sys.stderr)
        return 1
    if not np.isfinite(action).all():
        print("FAIL: non-finite actions.", file=sys.stderr)
        return 1

    print("PASS: X-WAM full-model ROCm smoke OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
