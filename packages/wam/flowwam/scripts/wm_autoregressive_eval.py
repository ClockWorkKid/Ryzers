# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""FlowWAM AUTOREGRESSIVE MULTI-CHUNK WORLD-MODEL ROLLOUT on WorldArena RoboTwin2.0, Strix Halo
(gfx1151).

NOTE ON NAMING (corrected): this is NOT closed-loop control. It is an open-loop-family, long-horizon
*world-model* stress test: FlowWAM (the world-modeling checkpoint) is chained over multiple chunks
where the last DREAMED frame anchors the next chunk (`rollout_generate`), so video-prediction errors
accumulate. There is NO controller and NO action feedback -- the robot is never driven by the model.
The genuine closed loop (model -> IDM action expert -> RoboTwin `env.step`) lives in
demos/demo_closedloop_robotwin.sh using the upstream FlowWAM flow-action server + robotwin_policy.

The action signal each chunk is the ground-truth action-derived optical flow (SAPIEN robot-only
render -> RAFT -> reversible codec) on our `simulation/robotwin` base.

Per episode we produce, vs the reference RoboTwin SAPIEN rollout (test_dataset/video/<ep>.mp4):
  * a slowed two-column video  [ REAL (sim GT) | DREAM (autoregressive WM rollout) ]  (rule 2.b),
  * a per-frame divergence curve (PSNR, + SSIM if skimage present) with chunk-handoff markers that
    reveal drift accumulation across autoregressive steps,
  * summary.json with mean/last-frame metrics + the full per-frame arrays.

World-model checkpoint only (SeedVR2 refiner deferred). Reuses the upstream call surface (rule 2.1).
Config via env mirrors open_loop_eval.py plus MAX_ROLLOUTS (default 3 -> multi-chunk rollout).
"""
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
    r = max(32, int(round(v / 32)) * 32)
    if r != v:
        print(f"[cfg] snapping dim {v} -> {r} (multiple of 32 for dual-stream)")
    return r


FLOWWAM_REPO = _env("FLOWWAM_REPO", "/repos/flowwam")
MODEL_DIR = _env("FLOWWAM_MODEL_DIR", "/models/flowwam")
CKPT = _env("FLOWWAM_CKPT", os.path.join(MODEL_DIR, "stage_1", "flowwam_worldarena_stage1.safetensors"))
DATA = _env("TEST_DATASET_DIR", "/data/test_dataset")
EMB = _env("EMBODIMENT_DIR", "/models/flowwam/embodiments")
OUT = _env("OUT_DIR", "/outputs/closedloop")
VARIANT = _env("VARIANT", "aloha-agilex_clean_50")
CAMERA = _env("CAMERA", "head_camera")

NUM_OUTPUT_FRAMES = _env("NUM_OUTPUT_FRAMES", 33, int)   # chunk size
NUM_STEPS = _env("NUM_STEPS", 15, int)
SIZE = (_mult32(_env("SIZE_W", 384, int)), _mult32(_env("SIZE_H", 288, int)))
FLOW_RES = (_env("FLOW_W", 384, int), _env("FLOW_H", 288, int))
MAX_ROLLOUTS = _env("MAX_ROLLOUTS", 3, int)              # >1 => closed loop on own frames
SIGMA_SHIFT = _env("SIGMA_SHIFT", 5.0, float)
SEED = _env("SEED", 1, int)
SLOW_FPS = _env("SLOW_FPS", 8, int)
FLOW_MAX_MAG = _env("FLOW_MAX_MAGNITUDE", 20.0, float)
MAX_EPISODES = _env("MAX_EPISODES", 5, int)
TILED = bool(_env("TILED", 0, int))
EPISODES = os.environ.get("EPISODES", "").replace(",", " ").split()

try:
    from skimage.metrics import structural_similarity as _ssim
    HAVE_SSIM = True
except Exception:  # noqa: BLE001
    HAVE_SSIM = False


def label(img, text):
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


def plot_divergence(psnr, ssim, boundaries, path, title):
    """Line plot of per-frame PSNR (and SSIM) with chunk-handoff markers. matplotlib if present,
    else a compact numpy/PIL fallback so the artifact is always produced."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax1 = plt.subplots(figsize=(7, 3.2))
        x = np.arange(len(psnr))
        ax1.plot(x, psnr, color="tab:blue", label="PSNR (dB)")
        ax1.set_xlabel("frame (autoregressive timeline)")
        ax1.set_ylabel("PSNR (dB)", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        if ssim is not None:
            ax2 = ax1.twinx()
            ax2.plot(x, ssim, color="tab:green", alpha=0.8, label="SSIM")
            ax2.set_ylabel("SSIM", color="tab:green")
            ax2.tick_params(axis="y", labelcolor="tab:green")
        for b in boundaries:
            ax1.axvline(b, color="tab:red", ls="--", lw=1, alpha=0.6)
        ax1.set_title(title)
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        return
    except Exception as e:  # noqa: BLE001
        print(f"  [plot] matplotlib unavailable ({e}); PIL fallback")
    W, H = 700, 300
    canvas = np.full((H, W, 3), 255, np.uint8)
    p = np.asarray(psnr, np.float32)
    lo, hi = float(p.min()), float(p.max() + 1e-6)
    for i in range(1, len(p)):
        x0 = int((i - 1) / max(1, len(p) - 1) * (W - 20)) + 10
        x1 = int(i / max(1, len(p) - 1) * (W - 20)) + 10
        y0 = int(H - 20 - (p[i - 1] - lo) / (hi - lo) * (H - 40))
        y1 = int(H - 20 - (p[i] - lo) / (hi - lo) * (H - 40))
        cv2.line(canvas, (x0, y0), (x1, y1), (200, 100, 0), 2)
    for b in boundaries:
        xb = int(b / max(1, len(p) - 1) * (W - 20)) + 10
        cv2.line(canvas, (xb, 10), (xb, H - 10), (0, 0, 220), 1)
    cv2.putText(canvas, f"{title}  PSNR {lo:.1f}-{hi:.1f}dB", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    Image.fromarray(canvas).save(path)


def main() -> int:
    import torch
    for p in (FLOWWAM_REPO, os.path.join(FLOWWAM_REPO, "inference")):
        if p not in sys.path:
            sys.path.insert(0, p)
    from world_model_inference import (
        RoboTwinRolloutInferenceDataset, build_pipeline, rollout_generate)
    from dataset_world_robotwin import add_bg_texture, mask_flows_by_robot

    class FastRolloutDataset(RoboTwinRolloutInferenceDataset):
        """Render only the frames actually used (identical output, ~20x less SAPIEN render)."""
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
            flat = self._build_flat_indices(T, num_rollouts)
            uniq = sorted(set(flat))
            rendered = self._robot_renderer.render_episode(sample["action_hdf5"], uniq, camera=self.camera)
            pos = {t: i for i, t in enumerate(uniq)}
            ro_sub = [rendered[pos[t]] for t in flat]
            if self.flow_resolution is not None:
                fw, fh = self.flow_resolution
                ro_flow = [cv2.resize(f, (fw, fh), interpolation=cv2.INTER_LINEAR) for f in ro_sub]
            else:
                ro_flow = ro_sub
            flows = self._compute_flows(add_bg_texture(ro_flow))
            flows = mask_flows_by_robot(flows, ro_flow)
            flows = [self._resize_flow(f) for f in flows]
            flow_pil, _ = self._encode_flows_to_pil(flows)
            flow_pil.insert(0, Image.fromarray(np.full((h, w, 3), 255, np.uint8)))
            return {"flow_video": flow_pil, "reference_image": first_frame, "prompt": prompt,
                    "episode_name": sample["episode_name"], "total_action_frames": T,
                    "num_rollouts": num_rollouts}

    print(f"torch {torch.__version__} hip={torch.version.hip} dev={torch.cuda.get_device_name(0)}")
    print(f"cfg: chunk={NUM_OUTPUT_FRAMES} steps={NUM_STEPS} size={SIZE} max_rollouts={MAX_ROLLOUTS} "
          f"tiled={TILED} ssim={HAVE_SSIM}")
    os.makedirs(OUT, exist_ok=True)
    device = torch.device("cuda")

    ds = FastRolloutDataset(
        test_dataset_dir=DATA, robot_only_dir=None, camera=CAMERA, size=SIZE,
        num_frames=NUM_OUTPUT_FRAMES, flow_method="raft", flow_device="cuda",
        flow_max_magnitude=FLOW_MAX_MAG, embodiment_dir=EMB, variant=VARIANT,
        instruction_variant=0, flow_resolution=FLOW_RES, robot_render_resolution=SIZE,
        max_stride=3, max_rollouts=MAX_ROLLOUTS)

    if EPISODES:
        want = list(dict.fromkeys(EPISODES))
        by_name = {s["episode_name"]: s for s in ds.samples}
        ds.samples = [by_name[n] for n in want if n in by_name]
    elif MAX_EPISODES:
        ds.samples = ds.samples[:MAX_EPISODES]
    print(f"episodes ({len(ds.samples)}): {[s['episode_name'] for s in ds.samples]}")

    t0 = time.time()
    pipe, flow_stream = build_pipeline(device, CKPT, MODEL_DIR)
    print(f"pipeline built in {time.time()-t0:.1f}s")

    # merge into an existing summary.json so per-episode processes (fresh CUDA context each, to
    # avoid cross-episode GPU-state corruption on the 32GB APU) accumulate into one file.
    summary_path = os.path.join(OUT, "summary.json")
    summary = {"config": {"chunk": NUM_OUTPUT_FRAMES, "steps": NUM_STEPS, "size": SIZE,
                          "max_rollouts": MAX_ROLLOUTS, "tiled": TILED}, "episodes": []}
    if os.path.exists(summary_path):
        try:
            with open(summary_path) as f:
                summary = json.load(f)
            summary.setdefault("episodes", [])
        except Exception:  # noqa: BLE001
            pass
    for i in range(len(ds)):
        batch = ds[i]
        ep = batch["episode_name"]
        nr = batch["num_rollouts"]
        print(f"\n===== {ep}: T={batch['total_action_frames']} num_rollouts={nr} "
              f"flow_frames={len(batch['flow_video'])} =====", flush=True)

        tg = time.time()
        dreamed = rollout_generate(
            pipe=pipe, flow_stream=flow_stream, prompt=batch["prompt"],
            initial_frame=batch["reference_image"], all_flow_frames=batch["flow_video"],
            num_rollouts=nr, chunk_size=NUM_OUTPUT_FRAMES, num_inference_steps=NUM_STEPS,
            sigma_shift=SIGMA_SHIFT, seed=SEED, tiled=TILED)
        gen_dt = time.time() - tg
        dream = [np.asarray(f, np.uint8) for f in dreamed]
        n = len(dream)
        print(f"  closed-loop rollout: {n} frames over {nr} chunk(s) in {gen_dt:.1f}s", flush=True)

        w, h = SIZE
        gt = load_gt_video(ep, n, SIZE)
        has_gt = gt is not None
        if not has_gt:
            print(f"  [warn] no GT video for {ep}; divergence metrics vs black frames.")
        # chunk handoff boundaries (rollout appends chunk[1:] after first chunk)
        boundaries = [NUM_OUTPUT_FRAMES + k * (NUM_OUTPUT_FRAMES - 1) for k in range(nr - 1)]

        psnr, ssim = [], ([] if HAVE_SSIM else None)
        panels = []
        for k in range(n):
            g = gt[k] if gt is not None else np.zeros((h, w, 3), np.uint8)
            d = dream[k]
            psnr.append(float(cv2.PSNR(g, d)))
            if HAVE_SSIM:
                ssim.append(float(_ssim(cv2.cvtColor(g, cv2.COLOR_RGB2GRAY),
                                        cv2.cvtColor(d, cv2.COLOR_RGB2GRAY))))
            panels.append(np.concatenate([label(g, "REAL (sim GT)"),
                                          label(d, "DREAM (autoregressive WM rollout)")], axis=1))

        imageio.mimwrite(os.path.join(OUT, f"{ep}_closedloop.mp4"), panels, fps=SLOW_FPS)
        Image.fromarray(panels[n // 2]).save(os.path.join(OUT, f"{ep}_closedloop_mid.png"))
        plot_divergence(psnr, ssim, boundaries, os.path.join(OUT, f"{ep}_divergence.png"),
                        f"{ep}: closed-loop dream vs real ({nr} chunks)")

        rec = {"episode": ep, "total_action_frames": int(batch["total_action_frames"]),
               "num_rollouts": int(nr), "frames": n, "has_gt": has_gt, "gen_seconds": round(gen_dt, 1),
               "psnr_mean": round(float(np.mean(psnr)), 3),
               "psnr_first": round(psnr[0], 3), "psnr_last": round(psnr[-1], 3),
               "psnr_per_frame": [round(v, 2) for v in psnr], "prompt": batch["prompt"]}
        if HAVE_SSIM:
            rec["ssim_mean"] = round(float(np.mean(ssim)), 4)
            rec["ssim_per_frame"] = [round(v, 3) for v in ssim]
        summary["episodes"] = [e for e in summary["episodes"] if e.get("episode") != ep]
        summary["episodes"].append(rec)
        print(f"  PSNR mean={rec['psnr_mean']}dB first={rec['psnr_first']} last={rec['psnr_last']} "
              f"(drift {rec['psnr_first']-rec['psnr_last']:+.2f}dB)"
              + (f"  SSIM={rec['ssim_mean']}" if HAVE_SSIM else ""), flush=True)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

    print(f"\nDONE. {len(summary['episodes'])} episodes -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
