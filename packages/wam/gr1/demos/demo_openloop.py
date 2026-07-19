# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""GR-1 open-loop prediction on a real CALVIN episode (no simulator).

Builds the real GR-1 policy exactly like upstream evaluation/calvin_evaluation.py (CLIP ViT-B/32
text encoder + MAE ViT-B image encoder + GR-1 snapshot), then replays a language-annotated CALVIN
episode window OFFLINE: at each step the model sees the true last `seq_len` observations and
predicts the next action. We compare the predicted 7-DoF action against the ground-truth
`rel_actions` (overlay plot, GT vs pred per rule 2.a) and, when the forward-prediction heads are
active, reconstruct a predicted future frame next to the ground-truth frame (two-column, rule 2.a).

This validates the full model graph on real data on Strix Halo (gfx1151) before wiring the
PyBullet simulator (phase 3). Run via demos/demo_openloop.sh (weights + data mounted).
"""
import argparse, json, os
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import clip
import models.vision_transformer as vits
from models.gr1 import GR1

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
ARM_LABELS = ["x", "y", "z", "roll", "pitch", "yaw"]


def build_policy(models_dir, mae_ckpt, policy_ckpt, configs_path, device):
    with open(configs_path) as f:
        variant = json.load(f)

    clip_root = os.path.join(models_dir, "clip_cache")
    os.makedirs(clip_root, exist_ok=True)
    model_clip, _ = clip.load(variant["clip_backbone"], device=device, download_root=clip_root)

    model_mae = vits.__dict__["vit_base"](patch_size=16, num_classes=0)
    mae_sd = torch.load(mae_ckpt, map_location="cpu")
    model_mae.load_state_dict(mae_sd["model"], strict=False)
    model_mae.to(device)

    training_target = [t for t in ("act_pred", "fwd_pred", "fwd_pred_hand") if variant.get(t)]
    policy = GR1(
        model_clip=model_clip, model_mae=model_mae,
        state_dim=variant["state_dim"], act_dim=variant["act_dim"],
        hidden_size=variant["embed_dim"], sequence_length=variant["seq_len"],
        training_target=training_target,
        img_feat_dim=variant["img_feat_dim"], patch_feat_dim=variant["patch_feat_dim"],
        lang_feat_dim=variant["lang_feat_dim"],
        resampler_params=dict(
            depth=variant["resampler_depth"], dim_head=variant["resampler_dim_head"],
            heads=variant["resampler_heads"], num_latents=variant["resampler_num_latents"],
            num_media_embeds=variant["resampler_num_media_embeds"]),
        without_norm_pixel_loss=variant["without_norm_pix_loss"], use_hand_rgb=variant["use_hand_rgb"],
        n_layer=variant["n_layer"], n_head=variant["n_head"], n_inner=4 * variant["embed_dim"],
        activation_function=variant["activation_function"], n_positions=variant["n_positions"],
        resid_pdrop=variant["dropout"], attn_pdrop=variant["dropout"])
    payload = torch.load(policy_ckpt, map_location="cpu")
    state_dict = payload["state_dict"] if "state_dict" in payload else payload
    msg = policy.load_state_dict(state_dict, strict=False)
    print(f"policy load: missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")
    policy.to(device).eval()
    return policy, variant


def unpatchify(patches, patch=16, grid=14):
    """(n_patches, p*p*3) -> (H, W, 3) inverse of GR-1's patchify permutation."""
    x = patches.reshape(grid, grid, patch, patch, 3)
    x = np.transpose(x, (0, 2, 1, 3, 4)).reshape(grid * patch, grid * patch, 3)
    return x


def norm01(a):
    a = a.astype(np.float32)
    lo, hi = a.min(), a.max()
    return (a - lo) / (hi - lo + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default=os.environ.get("MODELS_DIR", "/models"))
    ap.add_argument("--mae-ckpt", default=os.environ.get("MAE_CKPT", "/models/mae_pretrain_vit_base.pth"))
    ap.add_argument("--policy-ckpt", default=os.environ.get("POLICY_CKPT", "/models/snapshot_ABCD.pt"))
    ap.add_argument("--configs", default=os.environ.get("CONFIGS", "/repos/gr1/logs/configs.json"))
    ap.add_argument("--data-dir", default=os.environ.get("DATASET_DIR", "/data/calvin_debug_dataset"))
    ap.add_argument("--split", default="validation")
    ap.add_argument("--window-idx", type=int, default=0, help="index into the split's language annotations")
    ap.add_argument("--max-steps", type=int, default=32)
    ap.add_argument("--out", default=os.path.join(os.environ.get("OUT_DIR", "/outputs"), "openloop"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = "cuda"

    policy, variant = build_policy(args.models_dir, args.mae_ckpt, args.policy_ckpt, args.configs, device)
    seq_len, act_dim = variant["seq_len"], variant["act_dim"]

    split_dir = os.path.join(args.data_dir, args.split)
    ann = np.load(os.path.join(split_dir, "lang_annotations", "auto_lang_ann.npy"),
                  allow_pickle=True).item()
    langs = ann["language"]["ann"]
    indx = ann["info"]["indx"]
    wi = args.window_idx % len(langs)
    start, end = int(indx[wi][0]), int(indx[wi][1])
    lang = str(langs[wi])
    frames = list(range(start, min(end, start + args.max_steps - 1) + 1))
    print(f"window {wi}: '{lang}'  frames {start}..{frames[-1]} ({len(frames)} steps)")

    preprocess = T.Compose([T.Resize((224, 224), interpolation=Image.BICUBIC),
                            T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    tok = clip.tokenize([lang]).to(device)

    def load_rgb(arr):
        img = Image.fromarray(arr).convert("RGB")
        return preprocess(T.ToTensor()(img))

    rgb_buf, hand_buf, state_buf = deque(maxlen=seq_len), deque(maxlen=seq_len), deque(maxlen=seq_len)
    pred_arm, pred_grip, gt_arm, gt_grip = [], [], [], []
    fwd_saved = False

    for t in frames:
        ep = np.load(os.path.join(split_dir, f"episode_{t:07d}.npz"))
        rgb_buf.append(load_rgb(ep["rgb_static"]))
        hand_buf.append(load_rgb(ep["rgb_gripper"]))
        robot_obs = ep["robot_obs"].astype(np.float32)
        state_buf.append(torch.from_numpy(np.hstack([robot_obs[:6], robot_obs[-1]])).float())
        rel = ep["rel_actions"].astype(np.float32)

        n = len(rgb_buf)
        rgb_data = torch.zeros((1, seq_len, 3, 224, 224))
        hand_data = torch.zeros((1, seq_len, 3, 224, 224))
        rgb_data[0, :n] = torch.stack(list(rgb_buf))
        hand_data[0, :n] = torch.stack(list(hand_buf))
        st = torch.stack(list(state_buf))
        grip = -torch.ones((1, seq_len)); grip[0, :n] = st[:, 6]
        grip = ((grip + 1.0) / 2).long()
        grip = F.one_hot(grip, num_classes=2).float()
        arm = torch.zeros((1, seq_len, act_dim - 1)); arm[0, :n] = st[:, :6]
        attn = torch.zeros((1, seq_len), dtype=torch.long); attn[0, :n] = 1

        with torch.no_grad():
            pred = policy(rgb=rgb_data.to(device), hand_rgb=hand_data.to(device),
                          state={"arm": arm.to(device), "gripper": grip.to(device)},
                          language=tok, attention_mask=attn.to(device))
        i = n - 1
        pa = pred["arm_action_preds"][0, i].cpu().numpy()
        pg = float(torch.sigmoid(pred["gripper_action_preds"][0, i, 0]).cpu())
        pred_arm.append(pa); pred_grip.append(1.0 if pg > 0.5 else -1.0)
        gt_arm.append(rel[:6]); gt_grip.append(float(rel[6]))

        if not fwd_saved and pred.get("obs_preds") is not None and n == seq_len:
            op = pred["obs_preds"][0, i].cpu().numpy()  # (196, 768) normalized patches
            recon = norm01(unpatchify(op))
            gt_img = np.array(Image.fromarray(ep["rgb_static"]).convert("RGB").resize((224, 224)))
            fig, ax = plt.subplots(1, 2, figsize=(7, 3.6))
            ax[0].imshow(gt_img); ax[0].set_title("GT frame"); ax[0].axis("off")
            ax[1].imshow(recon); ax[1].set_title("GR-1 fwd pred (per-patch norm)"); ax[1].axis("off")
            fig.suptitle(f"'{lang}'  frame {t}")
            fig.tight_layout(); fig.savefig(os.path.join(args.out, "future_frame.png"), dpi=120)
            plt.close(fig); fwd_saved = True

    pred_arm = np.array(pred_arm); gt_arm = np.array(gt_arm)
    pred_grip = np.array(pred_grip); gt_grip = np.array(gt_grip)
    arm_mae = float(np.abs(pred_arm - gt_arm).mean())
    grip_acc = float((np.sign(pred_grip) == np.sign(gt_grip)).mean())
    print(f"open-loop arm MAE (rel action) : {arm_mae:.4f}")
    print(f"open-loop gripper match         : {grip_acc*100:.1f}%")

    steps = np.arange(len(pred_arm))
    fig, axes = plt.subplots(4, 2, figsize=(11, 10), sharex=True)
    for d in range(6):
        ax = axes[d // 2, d % 2]
        ax.plot(steps, gt_arm[:, d], "k-", label="GT", lw=1.6)
        ax.plot(steps, pred_arm[:, d], "r--", label="GR-1", lw=1.4)
        ax.set_title(f"arm {ARM_LABELS[d]}"); ax.grid(alpha=0.3)
        if d == 0:
            ax.legend(loc="upper right", fontsize=8)
    axg = axes[3, 0]
    axg.step(steps, gt_grip, "k-", where="mid", label="GT", lw=1.6)
    axg.step(steps, pred_grip, "r--", where="mid", label="GR-1", lw=1.4)
    axg.set_title("gripper (-1 close / +1 open)"); axg.set_ylim(-1.4, 1.4); axg.grid(alpha=0.3)
    axes[3, 1].axis("off")
    axes[3, 1].text(0.05, 0.6, f"task: '{lang}'\nsteps: {len(steps)}\n"
                    f"arm MAE: {arm_mae:.4f}\ngripper match: {grip_acc*100:.1f}%", fontsize=11)
    fig.suptitle("GR-1 open-loop action prediction vs CALVIN ground truth", fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "action_overlay.png"), dpi=120)
    plt.close(fig)

    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump({"window": wi, "task": lang, "steps": len(steps),
                   "arm_mae": arm_mae, "gripper_match": grip_acc}, f, indent=2)
    print(f"saved -> {args.out} (action_overlay.png, future_frame.png, metrics.json)")


if __name__ == "__main__":
    main()
