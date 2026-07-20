# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""AVDC open-loop video prediction on Strix Halo (gfx1151), no simulator.

Loads a real AVDC snapshot (+ CLIP ViT-B/32 text encoder) exactly like the upstream closed-loop
path (flowdiffusion.inference_utils.get_video_model / pred_video), feeds ONE initial frame + a
task string, and generates the predicted future video (8 frames = 1 conditioning + 7 predicted).
AVDC is actionless, so there are no ground-truth actions to score; the open-loop signal is the
predicted video. A reference frame exists, so per rule 2.b the initial frame is shown on the left
and the generated video on the right. Writes generated.gif, sidebyside.gif and metrics.json.
"""
import argparse, json, os, urllib.request

import numpy as np
from PIL import Image
import imageio.v2 as imageio

# Default example initial frame (upstream AVDC repo, fetched at runtime -> not re-hosted, rule 8).
DEFAULT_IMAGE_URL = "https://raw.githubusercontent.com/flow-diffusion/AVDC/main/examples/assembly.png"


def load_frame(path, size=128):
    """Match the upstream MW inference FoV: resize to 240x320 then center-crop `size`."""
    img = Image.open(path).convert("RGB").resize((320, 240))  # PIL (W, H)
    left, top = (320 - size) // 2, (240 - size) // 2
    img = img.crop((left, top, left + size, top + size))
    return np.asarray(img, dtype=np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default=os.environ.get("CKPT_DIR", "/models/metaworld"))
    ap.add_argument("--milestone", type=int, default=int(os.environ.get("MILESTONE") or "24"))
    ap.add_argument("--image", default=os.environ.get("IMAGE") or "")
    ap.add_argument("--text", default=os.environ.get("TEXT") or "assembly")
    ap.add_argument("--sample-steps", type=int, default=int(os.environ.get("SAMPLE_STEPS") or "100"))
    ap.add_argument("--flow", type=int, default=int(os.environ.get("FLOW") or "0"))
    ap.add_argument("--fps", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(os.environ.get("OUT_DIR", "/outputs"), "openloop"))
    ap.add_argument("--seed", type=int, default=int(os.environ.get("SEED") or "0"))
    args = ap.parse_args()

    import torch
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    from flowdiffusion.inference_utils import get_video_model, pred_video

    # Resolve the initial frame (download the upstream example if none supplied).
    img_path = args.image
    if not img_path:
        img_path = os.path.join(os.environ.get("OUT_DIR", "/outputs"), "assembly.png")
        if not os.path.isfile(img_path):
            print(f"fetching example frame -> {img_path}")
            urllib.request.urlretrieve(DEFAULT_IMAGE_URL, img_path)
    frame_0 = load_frame(img_path)
    print(f"initial frame    : {img_path}  {frame_0.shape}  task='{args.text}'")

    print(f"loading model    : {args.ckpt_dir}/model-{args.milestone}.pt (steps={args.sample_steps}, flow={args.flow})")
    model = get_video_model(ckpts_dir=args.ckpt_dir, milestone=args.milestone,
                            flow=bool(args.flow), timestep=args.sample_steps)
    import avdc_optim; avdc_optim.apply(model)   # env-gated fp16 / torch.compile (phase 4)

    preds = np.asarray(pred_video(model, frame_0, args.text, flow=bool(args.flow)))
    # upstream pred_video returns channels-first (T,3,H,W) on the RGB path (the ->HWC transpose
    # only runs in the flow branch); normalise to (T,H,W,3) uint8 for saving.
    if preds.ndim == 4 and preds.shape[1] == 3 and preds.shape[-1] != 3:
        preds = preds.transpose(0, 2, 3, 1)
    preds = preds.astype(np.uint8)
    n, h, w = preds.shape[0], preds.shape[1], preds.shape[2]
    print(f"generated video  : {preds.shape}  ({n} frames)")

    gen_gif = os.path.join(args.out, "generated.gif")
    imageio.mimsave(gen_gif, list(preds), duration=1000.0 / args.fps, loop=0)

    # rule 2.b: reference frame (left) | generated video (right), 2px separator.
    sep = np.full((h, 2, 3), 255, dtype=np.uint8)
    left = frame_0 if frame_0.shape[:2] == (h, w) else np.asarray(Image.fromarray(frame_0).resize((w, h)))
    combo = [np.concatenate([left, sep, preds[t]], axis=1) for t in range(n)]
    side_gif = os.path.join(args.out, "sidebyside.gif")
    imageio.mimsave(side_gif, combo, duration=1000.0 / args.fps, loop=0)

    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump({"task": args.text, "milestone": args.milestone, "sample_steps": args.sample_steps,
                   "flow": bool(args.flow), "n_frames": int(n), "frame_size": [int(h), int(w)],
                   "image": img_path}, f, indent=2)
    print(f"saved -> {args.out} (generated.gif, sidebyside.gif, metrics.json)")


if __name__ == "__main__":
    main()
