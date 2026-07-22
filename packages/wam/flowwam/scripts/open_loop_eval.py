# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""FlowWAM open-loop (action-driven) generation on the WorldArena RoboTwin2.0 test_dataset,
Strix Halo (gfx1151). Stage-1 only (the SeedVR2 refiner is deferred: apex/CUDA-only), so this is a
thin driver over the upstream call surface (rule 2.1): it reuses `RoboTwinRolloutInferenceDataset`
(SAPIEN robot-only render -> RAFT flow -> reversible codec), `build_pipeline`, and
`rollout_generate` from `world_model_inference.py`, and adds a GT-vs-dream two-column video
(rule 2.b; GT from test_dataset/video/<ep>.mp4) plus a flow-conditioning strip + summary.json.

Config via env (all optional): TEST_DATASET_DIR, EMBODIMENT_DIR, OUT_DIR, FLOWWAM_CKPT,
FLOWWAM_MODEL_DIR, EPISODES (space/comma list), MAX_EPISODES, NUM_OUTPUT_FRAMES, NUM_STEPS,
SIZE_W, SIZE_H, FLOW_W, FLOW_H, MAX_ROLLOUTS, SIGMA_SHIFT, SEED, FPS, FLOW_MAX_MAGNITUDE.
"""
import glob
import json
import os
import sys
import time

import numpy as np
import imageio
from PIL import Image


def _env(name, default, cast=str):
    v = os.environ.get(name)
    return cast(v) if v is not None and v != "" else default


FLOWWAM_REPO = _env("FLOWWAM_REPO", "/repos/flowwam")
MODEL_DIR = _env("FLOWWAM_MODEL_DIR", "/models/flowwam")
CKPT = _env("FLOWWAM_CKPT", os.path.join(MODEL_DIR, "stage_1", "flowwam_worldarena_stage1.safetensors"))
DATA = _env("TEST_DATASET_DIR", "/data/test_dataset")
EMB = _env("EMBODIMENT_DIR", "/models/flowwam/embodiments")
OUT = _env("OUT_DIR", "/outputs/openloop")
VARIANT = _env("VARIANT", "aloha-agilex_clean_50")
CAMERA = _env("CAMERA", "head_camera")

def _mult32(v):
    """Snap to a multiple of 32 so the VAE latent (v/16) is EVEN; otherwise the dual-stream
    rgb/flow token counts diverge (patchify floors odd latent rows -> shape mismatch)."""
    r = max(32, int(round(v / 32)) * 32)
    if r != v:
        print(f"[cfg] snapping dim {v} -> {r} (needs multiple of 32 for dual-stream)")
    return r


NUM_OUTPUT_FRAMES = _env("NUM_OUTPUT_FRAMES", 33, int)
NUM_STEPS = _env("NUM_STEPS", 15, int)
SIZE = (_mult32(_env("SIZE_W", 640, int)), _mult32(_env("SIZE_H", 480, int)))
FLOW_RES = (_env("FLOW_W", 320, int), _env("FLOW_H", 240, int))
MAX_ROLLOUTS = _env("MAX_ROLLOUTS", 1, int)
SIGMA_SHIFT = _env("SIGMA_SHIFT", 5.0, float)
SEED = _env("SEED", 1, int)
FPS = _env("FPS", 24, int)
FLOW_MAX_MAG = _env("FLOW_MAX_MAGNITUDE", 20.0, float)
MAX_EPISODES = _env("MAX_EPISODES", 1, int)
EPISODES = os.environ.get("EPISODES", "").replace(",", " ").split()


def load_gt_video(ep_name, n_target, size):
    """Load GT scene video, uniformly resample to n_target frames, resize to size (w,h)."""
    path = os.path.join(DATA, "video", f"{ep_name}.mp4")
    if not os.path.exists(path):
        return None
    rd = imageio.get_reader(path)
    frames = [f for f in rd]
    rd.close()
    if not frames:
        return None
    T = len(frames)
    idx = [round(i * (T - 1) / max(1, n_target - 1)) for i in range(n_target)]
    w, h = size
    out = []
    for i in idx:
        im = Image.fromarray(frames[i][:, :, :3]).resize((w, h), Image.BICUBIC)
        out.append(np.asarray(im, np.uint8))
    return out


def two_column(gt, dream, path, fps):
    """Write GT | dream side-by-side (rule 2.b). If gt is None, write dream alone."""
    if gt is None:
        imageio.mimwrite(path, list(dream), fps=fps)
        return False
    n = min(len(gt), len(dream))
    combo = [np.concatenate([gt[i], dream[i]], axis=1) for i in range(n)]
    imageio.mimwrite(path, combo, fps=fps)
    return True


def main() -> int:
    import torch
    for p in (FLOWWAM_REPO, os.path.join(FLOWWAM_REPO, "inference")):
        if p not in sys.path:
            sys.path.insert(0, p)
    from world_model_inference import (
        RoboTwinRolloutInferenceDataset, build_pipeline, rollout_generate)

    print(f"torch {torch.__version__} hip={torch.version.hip} dev={torch.cuda.get_device_name(0)}")
    print(f"cfg: frames={NUM_OUTPUT_FRAMES} steps={NUM_STEPS} size={SIZE} flow_res={FLOW_RES} "
          f"rollouts={MAX_ROLLOUTS} seed={SEED}")
    os.makedirs(OUT, exist_ok=True)
    device = torch.device("cuda")

    ds = RoboTwinRolloutInferenceDataset(
        test_dataset_dir=DATA, robot_only_dir=None, camera=CAMERA, size=SIZE,
        num_frames=NUM_OUTPUT_FRAMES, flow_method="raft", flow_device="cuda",
        flow_max_magnitude=FLOW_MAX_MAG, embodiment_dir=EMB, variant=VARIANT,
        instruction_variant=0, flow_resolution=FLOW_RES, robot_render_resolution=SIZE,
        max_stride=3, max_rollouts=MAX_ROLLOUTS)

    if EPISODES:
        ep_set = set(EPISODES)
        ds.samples = [s for s in ds.samples if s["episode_name"] in ep_set]
    elif MAX_EPISODES:
        ds.samples = ds.samples[:MAX_EPISODES]
    print(f"episodes to run: {[s['episode_name'] for s in ds.samples]}")

    t0 = time.time()
    pipe, flow_stream = build_pipeline(device, CKPT, MODEL_DIR)
    print(f"pipeline built in {time.time()-t0:.1f}s")

    summary = {"config": {"frames": NUM_OUTPUT_FRAMES, "steps": NUM_STEPS, "size": SIZE,
                          "flow_res": FLOW_RES, "rollouts": MAX_ROLLOUTS}, "episodes": []}
    for i in range(len(ds)):
        batch = ds[i]
        ep = batch["episode_name"]
        print(f"\n===== {ep}: T={batch['total_action_frames']} rollouts={batch['num_rollouts']} "
              f"flow_frames={len(batch['flow_video'])} =====", flush=True)
        print(f"  prompt: {batch['prompt'][:110]}...")

        # flow-conditioning strip (every ~1/8th) for inspection
        fv = batch["flow_video"]
        strip_idx = [round(k * (len(fv) - 1) / 7) for k in range(8)]
        strip = np.concatenate([np.asarray(fv[k].resize((160, 120))) for k in strip_idx], axis=1)
        Image.fromarray(strip).save(os.path.join(OUT, f"{ep}_flow_strip.png"))

        tg = time.time()
        dreamed = rollout_generate(
            pipe=pipe, flow_stream=flow_stream, prompt=batch["prompt"],
            initial_frame=batch["reference_image"], all_flow_frames=batch["flow_video"],
            num_rollouts=batch["num_rollouts"], chunk_size=NUM_OUTPUT_FRAMES,
            num_inference_steps=NUM_STEPS, sigma_shift=SIGMA_SHIFT, seed=SEED, tiled=True)
        gen_dt = time.time() - tg
        dream_np = [np.asarray(f, np.uint8) for f in dreamed]
        print(f"  generated {len(dream_np)} frames in {gen_dt:.1f}s "
              f"({gen_dt/max(1,batch['num_rollouts']*NUM_STEPS):.1f}s/step)", flush=True)

        imageio.mimwrite(os.path.join(OUT, f"{ep}_dream.mp4"), dream_np, fps=FPS)
        gt = load_gt_video(ep, len(dream_np), SIZE)
        had_gt = two_column(gt, dream_np, os.path.join(OUT, f"{ep}_gt_vs_dream.mp4"), FPS)
        Image.fromarray(dream_np[len(dream_np) // 2]).save(os.path.join(OUT, f"{ep}_dream_mid.png"))
        if had_gt:
            Image.fromarray(np.concatenate([gt[len(gt)//2], dream_np[len(dream_np)//2]], axis=1)
                            ).save(os.path.join(OUT, f"{ep}_gt_vs_dream_mid.png"))

        summary["episodes"].append({
            "episode": ep, "total_action_frames": int(batch["total_action_frames"]),
            "num_rollouts": int(batch["num_rollouts"]), "gen_frames": len(dream_np),
            "gen_seconds": round(gen_dt, 1), "has_gt": had_gt})

    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDONE. {len(summary['episodes'])} episodes -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
