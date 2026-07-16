# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Headless port of VERA's examples/droid_generation.ipynb for the Ryzer.

Runs the DROID WAN 14B video planner (i2v->v2v, action-free) on Strix Halo (gfx1151):
given a few real DROID context frames (3-cam canvas) + a language prompt, it "dreams" the
future frames. No server, no sim, no robot. Writes gen/context mp4s to $OUT_DIR.

Checkpoints (fetched by scripts/download_checkpoints.sh droid / vera_dl_droid.sh):
  VERA_DROID_CKPT_DIR   -> wan-droid-14b/ (algo_config.yaml + video_model.ckpt)
  VERA_WAN14B_CKPT_ROOT -> frozen Wan-AI/Wan2.1-I2V-14B-480P base (UMT5-XXL, VAE, CLIP)
  VERA_DROID_CLIPS_DIR  -> droid-demo-clips/ (bundled multi-view context clips)

Env knobs: SEED (default 11), MODE {scene1|scene2|both} (default both), TEXT (override the
scene-2 prompt with a single custom prompt), SCENE1_STARTS (csv of context start indices).
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

try:
    import av  # noqa: F401
except Exception as e:  # pragma: no cover
    print(f"FAIL: PyAV import failed: {e}", file=sys.stderr)
    raise
from PIL import Image

from vera.video_model.link.wan_pipeline import (
    WanPipeline,
    VideoCondition,
    GenerationConfig,
)
from vera.video_model.utils.video_utils import write_numpy_to_mp4

# --- config (mirrors the notebook) ---
CKPT_DIR = Path(os.environ.get("VERA_DROID_CKPT_DIR", "/models/vera-ckpts/wan-droid-14b"))
CLIPS_DIR = Path(os.environ.get("VERA_DROID_CLIPS_DIR", "/models/vera-ckpts/droid-demo-clips"))
OUT_DIR = Path(os.environ.get("OUT_DIR", "/outputs")) / "droid_generation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_H, MODEL_W_TOTAL, N_VIEWS = 128, 576, 3   # 3 views side by side on the canvas
VIEW_W = MODEL_W_TOTAL // N_VIEWS
TARGET_FPS = 15
SEED = int(os.environ.get("SEED", "11"))
MODE = os.environ.get("MODE", "both").lower()
CUSTOM_TEXT = os.environ.get("TEXT", "").strip()
SCENE1_STARTS = [int(x) for x in os.environ.get("SCENE1_STARTS", "0,10,20,30,40").split(",") if x != ""]


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
        print(f"  {name}: {frames.shape} @ {fps:.1f} fps -> {aligned[name].shape[0]} frames @ {TARGET_FPS}")
    n = min(v.shape[0] for v in aligned.values())
    resized = [np.stack([np.array(Image.fromarray(f).resize((VIEW_W, MODEL_H), Image.LANCZOS))
                         for f in aligned[name][:n]]) for name in order]
    return np.concatenate(resized, axis=2)   # [T, H, 3*W, 3] uint8


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


def save(full, name):
    path = OUT_DIR / name
    write_numpy_to_mp4(full, str(path), fps=TARGET_FPS)
    print(f"  wrote {path} ({full.shape[0]} frames)")


def _resolve(*cands):
    for c in cands:
        if Path(c).exists():
            return Path(c)
    return Path(cands[0])


def main() -> int:
    for var in ("VERA_WAN14B_CKPT_ROOT",):
        if not os.environ.get(var):
            print(f"FAIL: {var} not set (point it at the frozen Wan2.1-I2V-14B-480P dir).",
                  file=sys.stderr)
            return 2
    print(f"CKPT_DIR={CKPT_DIR}\nWAN14B={os.environ['VERA_WAN14B_CKPT_ROOT']}\nCLIPS={CLIPS_DIR}")
    print(f"device: {torch.cuda.get_device_name(0)}  seed={SEED}  mode={MODE}")

    pipeline = WanPipeline.from_config(
        config_path=str(CKPT_DIR / "algo_config.yaml"),
        ckpt_path=str(CKPT_DIR / "video_model.ckpt"),
        device="cuda:0",
        dtype=torch.bfloat16,
    )
    CTX = pipeline.required_pixel_frames
    CHUNK = pipeline.future_pixel_frames
    print(f"context={CTX} frames, generates {CHUNK} frames per call")

    if MODE in ("scene1", "both"):
        print("=== Scene 1 - rolling context ===")
        scene1 = stitch({
            "external": _resolve(CLIPS_DIR / "external_cam.mp4"),
            "wrist": _resolve(CLIPS_DIR / "wrist_cam.mp4"),
            "real": _resolve(CLIPS_DIR / "real.mov"),
        }, ["external", "real", "wrist"])
        prompt1 = CUSTOM_TEXT or ("a white robot arm reaches down to pick up a line of "
                                  "crackers and then places it into a white tray.")
        for start in SCENE1_STARTS:
            if start + CTX > scene1.shape[0]:
                print(f"  start={start}: not enough frames ({scene1.shape[0]}), skipping")
                continue
            torch.manual_seed(SEED + start)
            out = pipeline.generate(
                VideoCondition(context_frames=to_ctx(scene1[start:start + CTX]), text=prompt1),
                GenerationConfig(decode_outputs=["rgb"]),
            )
            pred = to_uint8(out["rgb"][0][CTX:])
            print(f"  start={start}: generated {pred.shape[0]} frames")
            save(np.concatenate([red_border(scene1[start:start + CTX].copy()), pred], axis=0),
                 f"scene1_start{start:03d}.mp4")

    if MODE in ("scene2", "both"):
        print("=== Scene 2 - language conditioning ===")
        ss = CLIPS_DIR / "second_set"
        scene2 = stitch({
            "varied_1": _resolve(ss / "varied_camera_1_35317039.mp4"),
            "varied_2": _resolve(ss / "varied_camera_2_39509833.mp4"),
            "hand": _resolve(ss / "hand_camera_16779706.mp4"),
        }, ["varied_1", "varied_2", "hand"])
        prompts2 = [CUSTOM_TEXT] if CUSTOM_TEXT else [
            "a white robot arm approaches the tennis ball. Then, its gripper closes on the tennis ball.",
            "a white robot arm approaches the yellow lego brick. Then, its gripper closes on the yellow lego brick.",
            "a white robot arm approaches the yellow teacup. Then, its gripper closes on the yellow teacup on its rim.",
        ]
        for i, prompt in enumerate(prompts2):
            torch.manual_seed(SEED + i)
            out = pipeline.generate(
                VideoCondition(context_frames=to_ctx(scene2[:CTX]), text=prompt),
                GenerationConfig(decode_outputs=["rgb"]),
            )
            pred = to_uint8(out["rgb"][0][CTX:])
            print(f"  prompt {i + 1}: {prompt}")
            save(np.concatenate([red_border(scene2[:CTX].copy()), pred], axis=0),
                 f"scene2_prompt{i + 1}.mp4")

    print("VIDEOGEN_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
