# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Imagined RGB + depth rollout generation for X-WAM on AMD Strix Halo (gfx1151).

X-WAM's headline capability over an action-only WAM is that its unified 4D world model
predicts a joint RGB *and depth* future: the DiT forks depth-modality branches
(`extra_blocks`/`extra_heads`) off the shared RGB backbone and reads out depth latents
alongside the video. This demo exercises that path end to end. Reusing the upstream
`RobotDataset` so the start observation matches training preprocessing (cf. FastWAM's
videogen_joint.py), it takes one ground-truth start frame (+ proprio + prompt) per clip and
runs `XWAMRunner.generate(early_stop=False, run_depth=True)`, which denoises the future RGB
latents, reads out the depth-modality latents, and VAE-decodes both. Each clip is written as
a GIF/MP4 stacking ground-truth RGB, the imagined RGB, the imagined depth, and the
ground-truth depth (turbo-colormapped), so the joint RGB+depth imagination is directly
visible; per-clip depth MAE vs GT is reported in imagination_summary.json.

Env: XWAM_REPO, CKPT_ROOT, WAN_CKPT_DIR, EXP (robotwin_sft|robocasa_sft), STEPS(last),
     DATASET_ROOT, DENOISE_STEPS(40), ACTION_DENOISE_STEPS(10), NUM_VIDEOS(2), CFG(0.0),
     SEED(0), OUT_DIR(/outputs), TAG, FPS(4).
"""
import os
import sys
import json
import time

import numpy as np
import torch

os.environ.setdefault("MPLBACKEND", "Agg")
import cv2
import imageio
from PIL import Image, ImageDraw

XWAM_REPO = os.environ.get("XWAM_REPO", "/repos/xwam")
CKPT_ROOT = os.environ.get("CKPT_ROOT", "/models/xwam/checkpoints")
WAN_CKPT_DIR = os.environ.get("WAN_CKPT_DIR", "/models/xwam/wan22_5b")
EXP = os.environ.get("EXP", "robotwin_sft")
STEPS = os.environ.get("STEPS", "last")
DATASET_ROOT = os.environ.get("DATASET_ROOT", "/models/xwam/datasets/RoboTwin")
DENOISE_STEPS = int(os.environ.get("DENOISE_STEPS") or "40")
ACTION_DENOISE_STEPS = int(os.environ.get("ACTION_DENOISE_STEPS") or "10")
NUM_VIDEOS = int(os.environ.get("NUM_VIDEOS") or "2")
CFG = float(os.environ.get("CFG") or "0.0")
SEED = int(os.environ.get("SEED") or "0")
OUT_DIR = os.environ.get("OUT_DIR", "/outputs")
TAG = os.environ.get("TAG", "robotwin")
FPS = int(os.environ.get("FPS") or "4")

_DATASET_KWARGS = ("sequence_length", "frame_skip", "action_skip", "video_size", "crop_ratio",
                   "brightness", "contrast", "saturation", "hue", "inverse_gripper",
                   "normalize_depths_per_view")


def _views_to_strip(frames_vtchw):
    """[V, T, C, H, W] float in [-1,1] -> uint8 [T, H, V*W, 3] (views concatenated horizontally)."""
    v, t = frames_vtchw.shape[0], frames_vtchw.shape[1]
    out = []
    for ti in range(t):
        views = []
        for vi in range(v):
            img = frames_vtchw[vi, ti].detach().float().clamp(-1, 1)
            img = ((img + 1) * 127.5).round().clamp(0, 255).byte().cpu().numpy()  # [C,H,W]
            views.append(np.transpose(img, (1, 2, 0)))  # [H,W,C]
        out.append(np.concatenate(views, axis=1))  # [H, V*W, C]
    return np.stack(out, axis=0)  # [T, H, V*W, 3]


def _colorize_depth(strip_thwc):
    """Grayscale depth strip [T,H,W,3] uint8 -> perceptual turbo colormap [T,H,W,3] uint8 (RGB)."""
    out = []
    for f in strip_thwc:
        gray = f.mean(axis=2).astype(np.uint8)
        cm = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)  # BGR
        out.append(cv2.cvtColor(cm, cv2.COLOR_BGR2RGB))
    return np.stack(out, axis=0)


def _label_rows(rows, labels, scale_w=720):
    """rows: list of [T,H,W,3] uint8 (same T,W). Stack vertically per frame, draw a left label
    banner on each row, and rescale the panel to <= scale_w wide. Returns [T, Hp, Wp, 3]."""
    T = min(r.shape[0] for r in rows)
    frames = []
    for ti in range(T):
        blocks = []
        for row, lab in zip(rows, labels):
            pil = Image.fromarray(row[ti].copy())
            d = ImageDraw.Draw(pil)
            d.rectangle([0, 0, max(150, 7 * len(lab) + 10), 16], fill=(0, 0, 0))
            d.text((4, 3), lab, fill=(255, 255, 0))
            blocks.append(np.array(pil))
        panel = np.concatenate(blocks, axis=0)  # [sum H, W, 3]
        h, w = panel.shape[:2]
        if w != scale_w:
            nh = int(round(h * scale_w / w))
            panel = np.array(Image.fromarray(panel).resize((scale_w, nh), Image.BILINEAR))
        frames.append(panel)
    return np.stack(frames, axis=0)


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
    if not config.get("use_depth", False):
        print("FAIL: this checkpoint was trained without depth (use_depth=false).", file=sys.stderr)
        return 1
    L.seed_everything(int(config.seed), workers=True)

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
    n_gen = min(NUM_VIDEOS, n_clips)
    idxs = np.linspace(0, n_clips - 1, n_gen, dtype=int).tolist()
    print(f"dataset          : {n_clips} clips; imagining {n_gen} at {idxs}")
    print(f"denoise          : video={DENOISE_STEPS}  action={ACTION_DENOISE_STEPS}  cfg={CFG}")

    t0 = time.time()
    model = XWAMRunner(config=config, run_depth=True).cuda().bfloat16()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["module"])
    model.eval()
    print(f"model loaded     : {time.time() - t0:.1f}s  params={sum(p.numel() for p in model.parameters())/1e9:.2f}B")

    os.makedirs(OUT_DIR, exist_ok=True)
    out_dir = os.path.join(OUT_DIR, f"imagination_{TAG}")
    os.makedirs(out_dir, exist_ok=True)

    per = []
    for k, i in enumerate(idxs):
        item = dataset[i]
        gt_video = item["video"]      # [V, T, C, H, W] in [-1,1]
        gt_depths = item["depths"]    # [V, T, C, H, W] in [-1,1]
        rgb0 = gt_video[:, 0].unsqueeze(0).bfloat16().cuda()           # [1, V, C, H, W]
        proprio0 = item["proprios"][0].unsqueeze(0).bfloat16().cuda()  # [1, Dp]
        prompt = item["prompt"]

        t1 = time.time()
        with torch.inference_mode():
            # pred_videos: uint8 [Tpix, (m*H), (V*W), 3] -- RGB rows on top, depth rows below.
            pred_videos, _, _, _ = model.generate(
                rgb0, proprio0, [prompt], seeds=[SEED],
                early_stop=False, cfg=CFG, run_depth=True)
        latency = time.time() - t1
        torch.cuda.empty_cache()

        Hm = pred_videos.shape[1] // 2
        pred_rgb = pred_videos[:, :Hm]          # [Tpix, H, V*W, 3] RGB
        pred_depth = pred_videos[:, Hm:]        # [Tpix, H, V*W, 3] depth (grayscale in RGB)

        gt_rgb = _views_to_strip(gt_video)      # [Tgt, H, V*W, 3]
        gt_depth = _views_to_strip(gt_depths)   # [Tgt, H, V*W, 3]
        tt = min(pred_rgb.shape[0], gt_rgb.shape[0])

        # Depth MAE (imagined vs GT) in [0,1] space over the aligned frames.
        pd = pred_depth[:tt].mean(axis=3).astype(np.float32) / 255.0
        gd = gt_depth[:tt].mean(axis=3).astype(np.float32) / 255.0
        depth_mae = float(np.abs(pd - gd).mean())

        rows = [gt_rgb[:tt], pred_rgb[:tt],
                _colorize_depth(pred_depth[:tt]), _colorize_depth(gt_depth[:tt])]
        labels = ["GT RGB (views)", "X-WAM imagined RGB",
                  "X-WAM imagined depth", "GT depth"]
        panel = _label_rows(rows, labels)

        gif = os.path.join(out_dir, f"imagination_{TAG}_clip{i}.gif")
        mp4 = os.path.join(out_dir, f"imagination_{TAG}_clip{i}.mp4")
        imageio.mimsave(gif, list(panel), fps=FPS, loop=0)
        imageio.mimwrite(mp4, list(panel), fps=FPS, quality=8, macro_block_size=1)
        per.append({"clip": int(i), "frames": int(tt), "latency_s": round(latency, 2),
                    "depth_mae_vs_gt": round(depth_mae, 4), "prompt": prompt[:80],
                    "gif": os.path.basename(gif)})
        print(f"  clip {i:>4} ({k+1}/{n_gen}): frames={tt} depthMAE={depth_mae:.4f} "
              f"{latency:.1f}s -> {os.path.basename(gif)}")

    summary = {"tag": TAG, "exp": EXP, "num_videos": n_gen, "denoise_steps": DENOISE_STEPS,
               "action_denoise_steps": ACTION_DENOISE_STEPS, "cfg": CFG, "clips": per}
    with open(os.path.join(out_dir, "imagination_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n== X-WAM {TAG} imagined RGB+depth rollout ==")
    print(f"clips + gifs + imagination_summary.json -> {out_dir}")
    print("PASS: X-WAM depth imagination complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
