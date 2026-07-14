# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""X-WAM RoboCasa VIDEO-DIFFUSION rollout harness (visualization, not benchmark).

The deployed policy runs X-WAM's cheap action-only path (early_stop, no VAE decode). This
harness instead exercises the *full* video-diffusion path (all sample_steps + the Wan2.2
multi-view RGB+depth VAE decode) so we can see what X-WAM actually imagines while planning.

For each episode it writes two videos, differing only in how the next plan is conditioned:
  * grounded   -- standard X-WAM cadence: re-observe the real sim every replan_steps and
                  plan from the fresh observation (the imagined video is re-anchored to
                  reality each chunk).
  * ungrounded -- open-loop world-model dream: after the first real observation, feed the
                  model its OWN predicted last frame + proprio as the next "observation"
                  (never re-grounding); the sim is still stepped with the predicted actions
                  only to provide a ground-truth reference of what those actions do.

Each frame is an upstream-style vertical stack (cf. xwam_runner.validation_step which cats
predictions over ground truth on dim=1): row0 = sim GT (3 views), row1 = X-WAM predicted RGB
(3 views), row2 = X-WAM predicted depth (3 views).

Env: TASK, NUM_EVALS, SEED_BASE, MAX_STEPS(0=default), REPLAN_STEPS(32), CFG(0.0),
MODES("grounded,ungrounded"), MAX_PLANS(12 cap per episode), OUT_DIR, FPS(6).
"""
import json
import os
import sys
from datetime import datetime

import numpy as np
from PIL import Image, ImageDraw

_EXP_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _EXP_DIR not in sys.path:
    sys.path.insert(0, _EXP_DIR)

from xwam_core import DirectXWAM, compute_seed  # noqa: E402
from sim_robocasa.render import font, save_mp4  # noqa: E402
from sim_robocasa.scene import build_scene  # noqa: E402

VW = 960  # montage width (3 views x 320); GT is resized to match
VH = 256  # per-row height


def _envi(k, d):
    v = os.environ.get(k)
    return int(v) if v not in (None, "") else d


def _resize(arr, w, h):
    return np.asarray(Image.fromarray(np.ascontiguousarray(arr)).resize((w, h), Image.BILINEAR))


def _label(img, text, color=(240, 240, 240)):
    im = Image.fromarray(np.ascontiguousarray(img))
    d = ImageDraw.Draw(im)
    f = font(18)
    d.rectangle([0, 0, 9 + 7 * len(text), 26], fill=(15, 15, 18))
    d.text((5, 3), text, fill=color, font=f, anchor="lt")
    return np.asarray(im)


def stack_frame(gt_view, pred_rgb_row, pred_depth_row, caption):
    """gt_view [Hg,Wg,3] uint8; pred rows [256,960,3] uint8 -> [3*256+, 960, 3] labeled."""
    gt = _resize(gt_view, VW, VH)
    rows = [
        _label(gt, "sim ground truth"),
        _label(pred_rgb_row, "X-WAM predicted RGB"),
        _label(pred_depth_row, "X-WAM predicted depth"),
    ]
    stacked = np.concatenate(rows, axis=0)
    banner = np.full((30, VW, 3), (20, 20, 24), np.uint8)
    banner = _label(banner, caption, color=(255, 200, 120))
    return np.concatenate([banner, stacked], axis=0)


def split_rgb_views(rgb_row):
    """[256, 960, 3] uint8 (3 views tiled) -> [-1,1] float32 [V=3, 256, 320, 3] for chaining."""
    v = rgb_row.reshape(256, 3, 320, 3).transpose(1, 0, 2, 3).astype(np.float32)
    return v / 127.5 - 1.0


def sample9(seq):
    """Pick 9 evenly-spaced items from a list (pads with last if short)."""
    if not seq:
        return []
    idx = np.linspace(0, len(seq) - 1, 9).round().astype(int)
    return [seq[i] for i in idx]


def run_episode_video(scene, model, instr, ep, seed, mode, replan_steps, max_steps, max_plans, cfg):
    real_obs = scene.reset()   # always tracks the true sim state (GT row)
    obs = real_obs             # what conditions the model (overwritten by prediction if ungrounded)
    frames = []
    plans = success = 0
    t = 0
    while t < max_steps and plans < max_plans:
        montage, actions, pred_prop = model.infer_video(
            obs["video"], obs["proprios"], [instr], seed=compute_seed(0, ep + 1, plans), cfg=cfg)
        n = min(replan_steps, len(actions))

        # step the sim with the predicted actions, collecting the REAL 3-view frame per step
        gt_views = [real_obs["view"]]
        for k in range(n):
            scene.step(actions[k])
            t += 1
            if scene.check_success():
                success = 1
            real_obs = scene.observe()
            gt_views.append(real_obs["view"])
            if success:
                break
        gt9 = sample9(gt_views)

        cap = f"{mode} | {instr[:70]} | plan {plans} step {t}"
        for j in range(montage.shape[0]):
            gt = gt9[min(j, len(gt9) - 1)]
            frames.append(stack_frame(gt, montage[j, 0:256], montage[j, 256:512], cap))

        plans += 1
        if success:
            break
        if mode == "ungrounded":
            # no re-grounding: next observation is the model's OWN last predicted frame+proprio
            obs = {"video": split_rgb_views(montage[-1, 0:256]),
                   "proprios": np.asarray(pred_prop[-1], dtype=np.float64)}
        else:
            obs = real_obs  # grounded: re-observe the real sim
    return frames, plans, bool(success)


def main():
    task = os.environ.get("TASK", "TurnOnSinkFaucet")
    num_evals = _envi("NUM_EVALS", 10)
    seed_base = _envi("SEED_BASE", 0)
    max_steps = _envi("MAX_STEPS", 0) or None
    replan_steps = _envi("REPLAN_STEPS", 32)
    max_plans = _envi("MAX_PLANS", 12)
    fps = _envi("FPS", 6)
    cfg = float(os.environ.get("CFG", "0.0"))
    modes = [m.strip() for m in os.environ.get("MODES", "grounded,ungrounded").split(",") if m.strip()]
    out_dir = os.environ.get("OUT_DIR", "/sim_outputs")
    save_dir = os.path.join(out_dir, "video_rollout", task)
    os.makedirs(save_dir, exist_ok=True)

    ckpt_root = os.environ.get("CKPT_ROOT", "/models/xwam/checkpoints")
    exp = os.environ.get("EXP", "robocasa_sft")
    model = DirectXWAM(
        exp_path=os.path.join(ckpt_root, exp),
        wan_checkpoint_dir=os.environ.get("WAN_CKPT_DIR", "/models/xwam/wan22_5b"),
        steps=os.environ.get("STEPS", "last"),
        denoise_steps=_envi("DENOISE_STEPS", 50),
        action_denoise_steps=_envi("ACTION_DENOISE_STEPS", 10),
    )
    print(f"[video_rollout] task={task} evals={num_evals} modes={modes} replan={replan_steps} "
          f"max_plans={max_plans} XWAM_OPT={os.environ.get('XWAM_OPT','0')}", flush=True)

    summary = []
    for i in range(num_evals):
        seed = seed_base + i
        for mode in modes:
            scene = build_scene(task, seed=seed)
            instr = scene.description
            t0 = datetime.now()
            frames, plans, success = run_episode_video(
                scene, model, instr, i, seed, mode, replan_steps, max_steps or scene.max_steps,
                max_plans, cfg)
            scene.close()
            dt = (datetime.now() - t0).total_seconds()
            tag = "success" if success else "run"
            path = os.path.join(save_dir, f"ep{i:02d}_seed{seed}_{mode}_{tag}.mp4")
            if frames:
                save_mp4(frames, path, fps=fps)
            print(f"[video_rollout] ep{i} seed={seed} {mode}: plans={plans} success={success} "
                  f"{dt:.0f}s -> {os.path.basename(path)}", flush=True)
            summary.append({"ep": i, "seed": seed, "mode": mode, "plans": plans,
                            "success": success, "seconds": round(dt, 1), "video": os.path.basename(path)})
            with open(os.path.join(save_dir, "_summary.json"), "w") as f:
                json.dump(summary, f, indent=2)

    print(f"[video_rollout] DONE {len(summary)} videos -> {save_dir}", flush=True)


if __name__ == "__main__":
    main()
