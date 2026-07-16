# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Same-scene, many-commands rollouts for the VERA DROID WAN 14B planner (ROCm).

Conditions the planner on ONE fixed starting scene (real DROID multi-view context) and rolls it
out under N different language commands with a FIXED seed, so the *prompt* is the only variable.
This isolates language-conditioning: identical context in, different dreamed future per command.

Writes $OUT_DIR/multiprompt/same_scene_p{i}.mp4 (red-bordered context + generated future) and
prints each command. Default 5 prompts target the objects in the second_set scene (tennis ball,
yellow lego brick, yellow teacup); override with PROMPTS (a '||'-separated list).

Env: checkpoint vars as in videogen_droid.py, plus SEED (default 11), PROMPTS (optional),
SCENE {second_set|first} (default second_set).
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
CLIPS_DIR = Path(os.environ.get("VERA_DROID_CLIPS_DIR", "/models/vera-ckpts/droid-demo-clips"))
OUT_DIR = Path(os.environ.get("OUT_DIR", "/outputs")) / "multiprompt"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_H, MODEL_W_TOTAL, N_VIEWS = 128, 576, 3
VIEW_W = MODEL_W_TOTAL // N_VIEWS
TARGET_FPS = 15
SEED = int(os.environ.get("SEED", "11"))
SCENE = os.environ.get("SCENE", "second_set").lower()

DEFAULT_PROMPTS = [
    "a white robot arm approaches the tennis ball. Then, its gripper closes on the tennis ball.",
    "a white robot arm approaches the yellow lego brick. Then, its gripper closes on the yellow lego brick.",
    "a white robot arm approaches the yellow teacup. Then, its gripper closes on the yellow teacup on its rim.",
    "a white robot arm approaches the tennis ball, closes its gripper, and lifts the tennis ball up.",
    "a white robot arm moves to the left and pushes the yellow lego brick across the table.",
]


def load_video_frames(path, max_frames=None):
    import av
    container = av.open(str(path))
    fps = float(container.streams.video[0].average_rate)
    frames = []
    for i, frame in enumerate(container.decode(video=0)):
        if max_frames is not None and i >= max_frames:
            break
        frames.append(frame.to_ndarray(format="rgb24"))
    container.close()
    return np.stack(frames), fps


def trim_and_resample(frames, src_fps, target_fps):
    ratio = src_fps / target_fps
    idx = [int(round(i * ratio)) for i in range(int(len(frames) / ratio))]
    return frames[[i for i in idx if i < len(frames)]]


def stitch(paths, order):
    aligned = {}
    for name, p in paths.items():
        frames, fps = load_video_frames(p)
        aligned[name] = trim_and_resample(frames, fps, TARGET_FPS)
    n = min(v.shape[0] for v in aligned.values())
    resized = [np.stack([np.array(Image.fromarray(f).resize((VIEW_W, MODEL_H), Image.LANCZOS))
                         for f in aligned[name][:n]]) for name in order]
    return np.concatenate(resized, axis=2)


def to_ctx(frames_uint8):
    t = torch.from_numpy(frames_uint8).float() / 127.5 - 1.0
    return t.permute(0, 3, 1, 2).unsqueeze(0)


def to_uint8(x):
    x = ((x.float() + 1.0) * 127.5).clamp(0, 255).round().to(torch.uint8)
    return x.permute(0, 2, 3, 1).cpu().numpy()


def red_border(frames, px=3):
    frames = frames.copy()
    frames[:, :px], frames[:, -px:] = (220, 30, 30), (220, 30, 30)
    frames[:, :, :px], frames[:, :, -px:] = (220, 30, 30), (220, 30, 30)
    return frames


def _resolve(*cands):
    for c in cands:
        if Path(c).exists():
            return Path(c)
    return Path(cands[0])


def build_context():
    if SCENE == "first":
        return stitch({
            "external": _resolve(CLIPS_DIR / "external_cam.mp4"),
            "wrist": _resolve(CLIPS_DIR / "wrist_cam.mp4"),
            "real": _resolve(CLIPS_DIR / "real.mov"),
        }, ["external", "real", "wrist"])
    ss = CLIPS_DIR / "second_set"
    return stitch({
        "varied_1": _resolve(ss / "varied_camera_1_35317039.mp4"),
        "varied_2": _resolve(ss / "varied_camera_2_39509833.mp4"),
        "hand": _resolve(ss / "hand_camera_16779706.mp4"),
    }, ["varied_1", "varied_2", "hand"])


def main() -> int:
    if not os.environ.get("VERA_WAN14B_CKPT_ROOT"):
        print("FAIL: VERA_WAN14B_CKPT_ROOT not set.", file=sys.stderr)
        return 2
    prompts = os.environ.get("PROMPTS", "").strip()
    prompts = [p.strip() for p in prompts.split("||") if p.strip()] if prompts else DEFAULT_PROMPTS
    print(f"device: {torch.cuda.get_device_name(0)}  seed={SEED} (fixed)  scene={SCENE}  "
          f"prompts={len(prompts)}", flush=True)

    pipeline = WanPipeline.from_config(
        config_path=str(CKPT_DIR / "algo_config.yaml"),
        ckpt_path=str(CKPT_DIR / "video_model.ckpt"),
        device="cuda:0",
        dtype=torch.bfloat16,
    )
    CTX = pipeline.required_pixel_frames
    CHUNK = pipeline.future_pixel_frames

    canvas = build_context()
    ctx_frames = canvas[:CTX]
    ctx = to_ctx(ctx_frames)
    print(f"context={CTX} frames -> generates {CHUNK} frames; canvas {canvas.shape}", flush=True)

    for i, prompt in enumerate(prompts, start=1):
        torch.manual_seed(SEED)  # fixed: prompt is the only variable
        out = pipeline.generate(
            VideoCondition(context_frames=ctx, text=prompt),
            GenerationConfig(decode_outputs=["rgb"]),
        )
        pred = to_uint8(out["rgb"][0][CTX:])
        full = np.concatenate([red_border(ctx_frames.copy()), pred], axis=0)
        path = OUT_DIR / f"same_scene_p{i}.mp4"
        write_numpy_to_mp4(full, str(path), fps=TARGET_FPS)
        print(f"  p{i}: {prompt}\n       wrote {path} ({full.shape[0]} frames)", flush=True)

    print("MULTIPROMPT_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
