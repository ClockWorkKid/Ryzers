# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Fixed-image + prompt -> video rollout demo for the VERA DROID WAN 14B planner (ROCm).

Unlike videogen_droid.py (which conditions on real DROID context *clips*), this lets a user
"start from a still frame": you give ONE image (or three per-camera images) plus a language
prompt, and the planner dreams a future rollout. The still is replicated across the model's
required context window (a static i2v prime), then the DiT generates the future frames.

Output (rule 2.b: reference image on the left, generated video on the right):
  $OUT_DIR/img2vid/<NAME>.mp4         side-by-side  [held input | generated rollout]
  $OUT_DIR/img2vid/<NAME>_gen.mp4     generated frames only

Env:
  IMAGE   path to a single starting image (treated as the full 3-view canvas, resized to 576x128)
  VIEWS   optional "ext.png,side.png,wrist.png" -> 3 per-camera images (overrides IMAGE)
  TEXT    the language prompt (required)
  NAME    output basename (default img2vid)
  SEED    (default 11)
Checkpoint env vars are identical to videogen_droid.py.
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from vera.video_model.link.wan_pipeline import (
    WanPipeline,
    VideoCondition,
    GenerationConfig,
)
from vera.video_model.utils.video_utils import write_numpy_to_mp4

CKPT_DIR = Path(os.environ.get("VERA_DROID_CKPT_DIR", "/models/vera-ckpts/wan-droid-14b"))
OUT_DIR = Path(os.environ.get("OUT_DIR", "/outputs")) / "img2vid"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_H, MODEL_W_TOTAL, N_VIEWS = 128, 576, 3
VIEW_W = MODEL_W_TOTAL // N_VIEWS
TARGET_FPS = 15
SEED = int(os.environ.get("PROF_SEED", os.environ.get("SEED", "11")))
NAME = os.environ.get("NAME", "img2vid")
TEXT = os.environ.get("TEXT", "").strip()


def load_canvas():
    """Build a [H, 3*W, 3] uint8 model canvas from IMAGE (whole canvas) or VIEWS (3 cams)."""
    views_env = os.environ.get("VIEWS", "").strip()
    if views_env:
        paths = [p.strip() for p in views_env.split(",") if p.strip()]
        if len(paths) != N_VIEWS:
            print(f"FAIL: VIEWS needs exactly {N_VIEWS} comma-separated image paths.", file=sys.stderr)
            raise SystemExit(2)
        tiles = [np.array(Image.open(p).convert("RGB").resize((VIEW_W, MODEL_H), Image.LANCZOS))
                 for p in paths]
        return np.concatenate(tiles, axis=1)
    img = os.environ.get("IMAGE", "").strip()
    if not img or not Path(img).exists():
        print(f"FAIL: set IMAGE to an existing image path (got {img!r}).", file=sys.stderr)
        raise SystemExit(2)
    return np.array(Image.open(img).convert("RGB").resize((MODEL_W_TOTAL, MODEL_H), Image.LANCZOS))


def to_ctx(frames_uint8):
    t = torch.from_numpy(frames_uint8).float() / 127.5 - 1.0
    return t.permute(0, 3, 1, 2).unsqueeze(0)


def to_uint8(x):
    x = ((x.float() + 1.0) * 127.5).clamp(0, 255).round().to(torch.uint8)
    return x.permute(0, 2, 3, 1).cpu().numpy()


def border(frames, color, px=3):
    frames = frames.copy()
    frames[:, :px], frames[:, -px:] = color, color
    frames[:, :, :px], frames[:, :, -px:] = color, color
    return frames


def main() -> int:
    if not os.environ.get("VERA_WAN14B_CKPT_ROOT"):
        print("FAIL: VERA_WAN14B_CKPT_ROOT not set.", file=sys.stderr)
        return 2
    if not TEXT:
        print("FAIL: set TEXT to a language prompt.", file=sys.stderr)
        return 2

    canvas = load_canvas()  # [H, 3W, 3]
    print(f"device: {torch.cuda.get_device_name(0)}  seed={SEED}\ninput canvas: {canvas.shape}\nprompt: {TEXT}",
          flush=True)

    pipeline = WanPipeline.from_config(
        config_path=str(CKPT_DIR / "algo_config.yaml"),
        ckpt_path=str(CKPT_DIR / "video_model.ckpt"),
        device="cuda:0",
        dtype=torch.bfloat16,
    )
    CTX = pipeline.required_pixel_frames
    CHUNK = pipeline.future_pixel_frames
    print(f"context={CTX} frames (static prime), generates {CHUNK} frames", flush=True)

    # replicate the still frame across the required context window
    static = np.repeat(canvas[None], CTX, axis=0)  # [CTX, H, 3W, 3]

    torch.manual_seed(SEED)
    out = pipeline.generate(
        VideoCondition(context_frames=to_ctx(static), text=TEXT),
        GenerationConfig(decode_outputs=["rgb"]),
    )
    gen = to_uint8(out["rgb"][0][CTX:])  # [CHUNK, H, 3W, 3]
    print(f"generated {gen.shape[0]} frames", flush=True)

    write_numpy_to_mp4(gen, str(OUT_DIR / f"{NAME}_gen.mp4"), fps=TARGET_FPS)

    # side-by-side: held input (left) | generated rollout (right)
    left = border(np.repeat(canvas[None], gen.shape[0], axis=0), (40, 120, 220))
    right = border(gen, (30, 200, 30))
    write_numpy_to_mp4(np.concatenate([left, right], axis=2), str(OUT_DIR / f"{NAME}.mp4"),
                       fps=TARGET_FPS)
    print(f"  wrote {OUT_DIR / (NAME + '.mp4')} and {OUT_DIR / (NAME + '_gen.mp4')}", flush=True)
    print("IMG2VID_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
