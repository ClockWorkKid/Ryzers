# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Open-loop replay for X-WAM on AMD Strix Halo (gfx1151).

Reuses the upstream dataset loader (`data.robot_dataset.RobotDataset`) to build model-ready
clips from released X-WAM episodes, then feeds the first observation of each clip through
`XWAMRunner.generate(..., early_stop=True)` and compares the predicted action chunk against
the ground-truth action chunk. Both are in the same quantile-normalized space (the dataset
normalizes GT; `generate` returns normalized predictions), so we report per-dimension
normalized MAE directly (cf. FastWAM open-loop) and overlay GT (solid) vs predicted (dashed)
on the same axes per rule 2.a.

Requires weights + a dataset subset (scripts/download_checkpoints.sh + download_datasets.sh).
Env: CKPT_ROOT, WAN_CKPT_DIR, EXP (robotwin_sft), DATASET_ROOT (/models/xwam/datasets/RoboTwin),
     DENOISE_STEPS=50, ACTION_DENOISE_STEPS=10, NUM_EPISODES=5, OUT_DIR=/outputs, TAG=robotwin.
"""
import os
import sys
import time

import numpy as np
import torch

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt

XWAM_REPO = os.environ.get("XWAM_REPO", "/repos/xwam")
CKPT_ROOT = os.environ.get("CKPT_ROOT", "/models/xwam/checkpoints")
WAN_CKPT_DIR = os.environ.get("WAN_CKPT_DIR", "/models/xwam/wan22_5b")
EXP = os.environ.get("EXP", "robotwin_sft")
STEPS = os.environ.get("STEPS", "last")
DATASET_ROOT = os.environ.get("DATASET_ROOT", "/models/xwam/datasets/RoboTwin")
DENOISE_STEPS = int(os.environ.get("DENOISE_STEPS") or "50")
ACTION_DENOISE_STEPS = int(os.environ.get("ACTION_DENOISE_STEPS") or "10")
NUM_EPISODES = int(os.environ.get("NUM_EPISODES") or "5")
CFG = float(os.environ.get("CFG") or "0.0")
OUT_DIR = os.environ.get("OUT_DIR", "/outputs")
TAG = os.environ.get("TAG", "robotwin")

ACTION_LABELS = ["L_x", "L_y", "L_z", "L_ax", "L_ay", "L_az", "L_grip",
                 "R_x", "R_y", "R_z", "R_ax", "R_ay", "R_az", "R_grip"]

_DATASET_KWARGS = ("sequence_length", "frame_skip", "action_skip", "video_size", "crop_ratio",
                   "brightness", "contrast", "saturation", "hue", "inverse_gripper",
                   "normalize_depths_per_view")


def main() -> int:
    print(f"torch            : {torch.__version__}  hip={torch.version.hip}")
    if not (torch.version.hip and torch.cuda.is_available()):
        print("FAIL: need a ROCm torch + visible GPU.", file=sys.stderr)
        return 1
    print(f"device[0]        : {torch.cuda.get_device_name(0)}")
    if XWAM_REPO not in sys.path:
        sys.path.insert(0, XWAM_REPO)

    from omegaconf import OmegaConf
    import lightning as L
    from runners.xwam_runner import XWAMRunner
    from data.robot_dataset import RobotDataset

    exp_path = os.path.join(CKPT_ROOT, EXP)
    cfg_path = os.path.join(exp_path, "config.yaml")
    ckpt_path = os.path.join(exp_path, f"checkpoints/{STEPS}.ckpt/checkpoint/mp_rank_00_model_states.pt")
    for p in (cfg_path, ckpt_path):
        if not os.path.exists(p):
            print(f"FAIL: not found: {p}", file=sys.stderr)
            return 1
    if not os.path.isdir(DATASET_ROOT):
        print(f"FAIL: dataset not found: {DATASET_ROOT}\n      run download_datasets.sh first.", file=sys.stderr)
        return 1

    config = OmegaConf.load(cfg_path)
    config.sample_steps = DENOISE_STEPS
    config.use_decoupled_inference = ACTION_DENOISE_STEPS > 0
    config.action_denoise_steps = ACTION_DENOISE_STEPS
    config.action_num = config.dataset.frame_skip // config.dataset.action_skip
    config.wan_checkpoint_dir = WAN_CKPT_DIR
    L.seed_everything(int(config.seed), workers=True)

    # Build the dataset with the SFT config's data params (no augmentation for eval).
    ds_src = config.dataset
    ds_kwargs = {k: OmegaConf.to_container(ds_src[k], resolve=True) if OmegaConf.is_config(ds_src[k])
                 else ds_src[k] for k in _DATASET_KWARGS if k in ds_src}
    dataset = RobotDataset(
        dataset_path=DATASET_ROOT,
        augment=False,
        shuffle_view_order=False,
        statistics=OmegaConf.to_container(ds_src.statistics, resolve=True),
        **ds_kwargs,
    )
    n_clips = len(dataset)
    n_eval = min(NUM_EPISODES, n_clips)
    idxs = np.linspace(0, n_clips - 1, n_eval, dtype=int).tolist()
    print(f"dataset          : {n_clips} clips; evaluating {n_eval} at {idxs}")

    t0 = time.time()
    model = XWAMRunner(config=config).cuda().bfloat16()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["module"])
    model.eval()
    print(f"model loaded     : {time.time() - t0:.1f}s  params={sum(p.numel() for p in model.parameters())/1e9:.2f}B")

    per_dim_abs, first = [], None
    for k, i in enumerate(idxs):
        item = dataset[i]
        rgb0 = item["video"][:, 0].unsqueeze(0).bfloat16().cuda()      # [1, V, C, H, W]
        proprio0 = item["proprios"][0].unsqueeze(0).bfloat16().cuda()  # [1, Dp]
        gt = item["actions"].float().numpy()                           # [Ta, ad] normalized
        mask = item["action_mask"].float().numpy()                     # [Ta, ad]
        prompt = item["prompt"]

        tS = time.time()
        with torch.inference_mode():
            _, xt_actions, _, _ = model.generate(
                rgb0, proprio0, [prompt], seeds=[0], early_stop=True, cfg=CFG, run_depth=False)
        pred = xt_actions[0].float().cpu().numpy()                     # [Ta, ad] normalized
        abs_err = np.abs(pred - gt) * mask
        per_dim_abs.append(abs_err.sum(axis=0) / np.maximum(mask.sum(axis=0), 1e-6))
        mae = abs_err.sum() / max(mask.sum(), 1e-6)
        print(f"  clip {i:>4} ({k+1}/{n_eval}): MAE={mae:.4f}  {time.time()-tS:.1f}s  \"{prompt[:48]}\"")
        if first is None:
            first = (i, gt, pred, mask, prompt)

    per_dim = np.mean(np.stack(per_dim_abs), axis=0)   # [ad]
    overall = float(per_dim.mean())
    print(f"\nmean normalized MAE : {overall:.4f}  (over {n_eval} clips)")
    for lbl, v in zip(ACTION_LABELS[:len(per_dim)], per_dim):
        print(f"    {lbl:<6}: {v:.4f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    # Plot 1: per-dim normalized MAE bar.
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.bar(range(len(per_dim)), per_dim, color="#2b6cb0")
    ax.set_xticks(range(len(per_dim))); ax.set_xticklabels(ACTION_LABELS[:len(per_dim)], rotation=45, ha="right")
    ax.set_ylabel("normalized MAE"); ax.set_title(f"X-WAM {TAG} open-loop per-dim MAE (mean {overall:.4f}, {n_eval} clips)")
    fig.tight_layout(); p1 = os.path.join(OUT_DIR, f"openloop_{TAG}_per_dim_mae.png")
    fig.savefig(p1, dpi=100); plt.close(fig)

    # Plot 2: GT (solid) vs predicted (dashed) overlay for the first clip (rule 2.a).
    i0, gt0, pred0, mask0, prompt0 = first
    ad = gt0.shape[1]; ncol = 4; nrow = int(np.ceil(ad / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3 * ncol, 2.1 * nrow), squeeze=False)
    for d in range(nrow * ncol):
        ax = axes[d // ncol][d % ncol]
        if d < ad:
            ax.plot(gt0[:, d], "-", color="#1a7f37", label="GT")
            ax.plot(pred0[:, d], "--", color="#cf222e", label="pred")
            ax.set_title(ACTION_LABELS[d], fontsize=9); ax.tick_params(labelsize=7)
        else:
            ax.axis("off")
    axes[0][0].legend(fontsize=8, loc="best")
    fig.suptitle(f"X-WAM {TAG} open-loop GT vs pred (clip {i0}): \"{prompt0[:60]}\"", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); p2 = os.path.join(OUT_DIR, f"openloop_{TAG}_clip{i0}_overlay.png")
    fig.savefig(p2, dpi=100); plt.close(fig)

    print(f"saved            : {p1}\n                   {p2}")
    if not np.isfinite(overall):
        print("FAIL: non-finite MAE.", file=sys.stderr); return 1
    print("PASS: X-WAM open-loop replay OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
