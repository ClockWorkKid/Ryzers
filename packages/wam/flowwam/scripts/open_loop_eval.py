# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""FlowWAM open-loop (action-driven) generation on the WorldArena RoboTwin2.0 test_dataset,
Strix Halo (gfx1151). Stage-1 only (the SeedVR2 refiner is deferred: apex/CUDA-only), so this is a
thin driver over the upstream call surface (rule 2.1): it reuses `RoboTwinRolloutInferenceDataset`
(SAPIEN robot-only render -> RAFT flow -> reversible codec), `build_pipeline`, and
`rollout_generate` from `world_model_inference.py`.

Primary artifact per episode: a THREE-PANEL, slowed-down video that makes the model legible ---
    [ REAL OBS (GT) | FLOW FIELD (action conditioning) | DREAM (generated) ]
GT is uniformly resampled to align frame-by-frame with the dream; the flow panel is the reversible
flow-codec image actually fed to the model. A mid-frame PNG + summary.json are also written.

Config via env (all optional): TEST_DATASET_DIR, EMBODIMENT_DIR, OUT_DIR, FLOWWAM_CKPT,
FLOWWAM_MODEL_DIR, EPISODES (space/comma list), MAX_EPISODES, NUM_OUTPUT_FRAMES, NUM_STEPS,
SIZE_W, SIZE_H, FLOW_W, FLOW_H, MAX_ROLLOUTS, SIGMA_SHIFT, SEED, SLOW_FPS, FLOW_MAX_MAGNITUDE,
TILED (1/0), FAST_RENDER (1/0: render only the frames actually used).
"""
import glob
import json
import math
import os
import sys
import time

import cv2
import numpy as np
import imageio
from PIL import Image


def _env(name, default, cast=str):
    v = os.environ.get(name)
    return cast(v) if v is not None and v != "" else default


def _mult32(v):
    """Snap to a multiple of 32 so the VAE latent (v/16) is EVEN; otherwise the dual-stream
    rgb/flow token counts diverge (patchify floors odd latent rows -> shape mismatch)."""
    r = max(32, int(round(v / 32)) * 32)
    if r != v:
        print(f"[cfg] snapping dim {v} -> {r} (needs multiple of 32 for dual-stream)")
    return r


FLOWWAM_REPO = _env("FLOWWAM_REPO", "/repos/flowwam")
MODEL_DIR = _env("FLOWWAM_MODEL_DIR", "/models/flowwam")
CKPT = _env("FLOWWAM_CKPT", os.path.join(MODEL_DIR, "stage_1", "flowwam_worldarena_stage1.safetensors"))
DATA = _env("TEST_DATASET_DIR", "/data/test_dataset")
EMB = _env("EMBODIMENT_DIR", "/models/flowwam/embodiments")
OUT = _env("OUT_DIR", "/outputs/openloop")
VARIANT = _env("VARIANT", "aloha-agilex_clean_50")
CAMERA = _env("CAMERA", "head_camera")

NUM_OUTPUT_FRAMES = _env("NUM_OUTPUT_FRAMES", 33, int)
NUM_STEPS = _env("NUM_STEPS", 15, int)
SIZE = (_mult32(_env("SIZE_W", 384, int)), _mult32(_env("SIZE_H", 288, int)))
FLOW_RES = (_env("FLOW_W", 384, int), _env("FLOW_H", 288, int))
MAX_ROLLOUTS = _env("MAX_ROLLOUTS", 1, int)
SIGMA_SHIFT = _env("SIGMA_SHIFT", 5.0, float)
SEED = _env("SEED", 1, int)
SLOW_FPS = _env("SLOW_FPS", 8, int)
FLOW_MAX_MAG = _env("FLOW_MAX_MAGNITUDE", 20.0, float)
MAX_EPISODES = _env("MAX_EPISODES", 1, int)
TILED = bool(_env("TILED", 1, int))
FAST_RENDER = bool(_env("FAST_RENDER", 1, int))
EPISODES = os.environ.get("EPISODES", "").replace(",", " ").split()


def label(img, text):
    """Draw a readable caption (black outline + white text) on a copy of img."""
    img = np.ascontiguousarray(img.copy())
    cv2.putText(img, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def resample_list(frames, n):
    T = len(frames)
    if T == n or T == 0:
        return frames
    idx = [round(i * (T - 1) / max(1, n - 1)) for i in range(n)]
    return [frames[i] for i in idx]


def load_gt_video(ep_name, n_target, size):
    path = os.path.join(DATA, "video", f"{ep_name}.mp4")
    if not os.path.exists(path):
        return None
    rd = imageio.get_reader(path)
    frames = [f for f in rd]
    rd.close()
    if not frames:
        return None
    w, h = size
    return [np.asarray(Image.fromarray(f[:, :, :3]).resize((w, h), Image.BICUBIC), np.uint8)
            for f in resample_list(frames, n_target)]


def main() -> int:
    import torch
    for p in (FLOWWAM_REPO, os.path.join(FLOWWAM_REPO, "inference")):
        if p not in sys.path:
            sys.path.insert(0, p)
    from world_model_inference import (
        RoboTwinRolloutInferenceDataset, build_pipeline, rollout_generate)
    from dataset_world_robotwin import add_bg_texture, mask_flows_by_robot

    # --- pure-optimization subclass: render ONLY the frames actually used (identical output) ---
    class FastRolloutDataset(RoboTwinRolloutInferenceDataset):
        def __getitem__(self, idx):
            sample = self.samples[idx]
            w, h = self.size
            first_frame = Image.open(sample["first_frame_path"]).convert("RGB").resize((w, h), Image.BICUBIC)
            prompt = ""
            if os.path.exists(sample["instruction_path"]):
                with open(sample["instruction_path"], "r") as f:
                    prompt = json.load(f).get("instruction", "")

            self._ensure_robot_renderer()
            T = self._robot_renderer.get_episode_length(sample["action_hdf5"])
            chunk_interval = (self.num_frames - 1) * self.max_stride
            num_rollouts = min(self.max_rollouts, max(1, math.ceil((T - 1) / chunk_interval)))
            flat_indices = self._build_flat_indices(T, num_rollouts)

            uniq = sorted(set(flat_indices))
            rendered = self._robot_renderer.render_episode(sample["action_hdf5"], uniq, camera=self.camera)
            pos = {t: i for i, t in enumerate(uniq)}
            ro_sub = [rendered[pos[t]] for t in flat_indices]

            if self.flow_resolution is not None:
                fw, fh = self.flow_resolution
                ro_flow = [cv2.resize(f, (fw, fh), interpolation=cv2.INTER_LINEAR) for f in ro_sub]
            else:
                ro_flow = ro_sub
            textured = add_bg_texture(ro_flow)
            flows = self._compute_flows(textured)
            flows = mask_flows_by_robot(flows, ro_flow)
            flows = [self._resize_flow(f) for f in flows]
            flow_pil, _ = self._encode_flows_to_pil(flows)
            flow_pil.insert(0, Image.fromarray(np.full((h, w, 3), 255, np.uint8)))
            return {"flow_video": flow_pil, "reference_image": first_frame, "prompt": prompt,
                    "episode_name": sample["episode_name"], "total_action_frames": T,
                    "num_rollouts": num_rollouts}

    print(f"torch {torch.__version__} hip={torch.version.hip} dev={torch.cuda.get_device_name(0)}")
    print(f"cfg: frames={NUM_OUTPUT_FRAMES} steps={NUM_STEPS} size={SIZE} tiled={TILED} "
          f"fast_render={FAST_RENDER} slow_fps={SLOW_FPS}")
    os.makedirs(OUT, exist_ok=True)
    device = torch.device("cuda")

    ds_cls = FastRolloutDataset if FAST_RENDER else RoboTwinRolloutInferenceDataset
    ds = ds_cls(
        test_dataset_dir=DATA, robot_only_dir=None, camera=CAMERA, size=SIZE,
        num_frames=NUM_OUTPUT_FRAMES, flow_method="raft", flow_device="cuda",
        flow_max_magnitude=FLOW_MAX_MAG, embodiment_dir=EMB, variant=VARIANT,
        instruction_variant=0, flow_resolution=FLOW_RES, robot_render_resolution=SIZE,
        max_stride=3, max_rollouts=MAX_ROLLOUTS)

    if EPISODES:
        want = list(dict.fromkeys(EPISODES))
        by_name = {s["episode_name"]: s for s in ds.samples}
        ds.samples = [by_name[n] for n in want if n in by_name]
        missing = [n for n in want if n not in by_name]
        if missing:
            print(f"[warn] episodes not found: {missing}")
    elif MAX_EPISODES:
        ds.samples = ds.samples[:MAX_EPISODES]
    print(f"episodes to run ({len(ds.samples)}): {[s['episode_name'] for s in ds.samples]}")

    t0 = time.time()
    pipe, flow_stream = build_pipeline(device, CKPT, MODEL_DIR)
    print(f"pipeline built in {time.time()-t0:.1f}s")

    summary = {"config": {"frames": NUM_OUTPUT_FRAMES, "steps": NUM_STEPS, "size": SIZE,
                          "flow_res": FLOW_RES, "rollouts": MAX_ROLLOUTS, "tiled": TILED,
                          "slow_fps": SLOW_FPS}, "episodes": []}
    for i in range(len(ds)):
        batch = ds[i]
        ep = batch["episode_name"]
        print(f"\n===== {ep}: T={batch['total_action_frames']} rollouts={batch['num_rollouts']} "
              f"flow_frames={len(batch['flow_video'])} =====", flush=True)
        print(f"  prompt: {batch['prompt'][:110]}...")

        tg = time.time()
        dreamed = rollout_generate(
            pipe=pipe, flow_stream=flow_stream, prompt=batch["prompt"],
            initial_frame=batch["reference_image"], all_flow_frames=batch["flow_video"],
            num_rollouts=batch["num_rollouts"], chunk_size=NUM_OUTPUT_FRAMES,
            num_inference_steps=NUM_STEPS, sigma_shift=SIGMA_SHIFT, seed=SEED, tiled=TILED)
        gen_dt = time.time() - tg
        dream_np = [np.asarray(f, np.uint8) for f in dreamed]
        n = len(dream_np)
        print(f"  generated {n} frames in {gen_dt:.1f}s", flush=True)

        # align GT + flow to the dream length, then build the labelled three-panel
        w, h = SIZE
        gt = load_gt_video(ep, n, SIZE)
        flow_imgs = [np.asarray(f.convert("RGB").resize((w, h)), np.uint8) for f in batch["flow_video"]]
        flow_imgs = resample_list(flow_imgs, n)
        gt = gt if gt is not None else [np.zeros((h, w, 3), np.uint8)] * n

        panels = []
        for k in range(n):
            row = np.concatenate([label(gt[k], "REAL OBS"),
                                  label(flow_imgs[k], "FLOW (action)"),
                                  label(dream_np[k], "DREAM")], axis=1)
            panels.append(row)
        imageio.mimwrite(os.path.join(OUT, f"{ep}_three_panel.mp4"), panels, fps=SLOW_FPS)
        Image.fromarray(panels[n // 2]).save(os.path.join(OUT, f"{ep}_three_panel_mid.png"))

        summary["episodes"].append({
            "episode": ep, "total_action_frames": int(batch["total_action_frames"]),
            "num_rollouts": int(batch["num_rollouts"]), "gen_frames": n,
            "gen_seconds": round(gen_dt, 1), "has_gt": gt is not None,
            "prompt": batch["prompt"]})
        with open(os.path.join(OUT, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

    print(f"\nDONE. {len(summary['episodes'])} episodes -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
