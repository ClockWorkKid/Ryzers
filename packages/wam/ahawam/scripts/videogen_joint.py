# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Open-loop imagined-video generation for AHA-WAM on Strix Halo (gfx1151).

Imagines the future clip from a ground-truth start frame + prompt and writes a
side-by-side GT-vs-imagined MP4. AHA-WAM's async runtime only prefills the video context
as a single-frame KV cache for the action executor, so imagination denoises the full
future directly with the shared Wan2.2 video expert (per-frame causal, first frame pinned
to the encoded observation). The video branch is not action-conditioned, so this
reproduces the model's imagined future. Reuses upstream RobotVideoDataset + AHAWAMProcessor
so preprocessing matches training.

Env: AHAWAM_REPO, CONFIG_NAME(sim_robotwin), CKPT, DATASET_STATS, DATASET_DIR,
     NUM_VIDEOS(10), NUM_STEPS(20), OUT_DIR, TAG, SEED, FPS(6).
"""
import os
import sys
import json
import time

import numpy as np
import torch
import imageio
from PIL import Image, ImageDraw

AHAWAM_REPO = os.environ.get("AHAWAM_REPO", "/repos/ahawam")
CONFIG_NAME = os.environ.get("CONFIG_NAME", "sim_robotwin")
CKPT = os.environ["CKPT"]
DATASET_STATS = os.environ["DATASET_STATS"]
DATASET_DIR = os.environ["DATASET_DIR"]
NUM_VIDEOS = int(os.environ.get("NUM_VIDEOS") or "10")
NUM_STEPS = int(os.environ.get("NUM_STEPS") or "20")
OUT_DIR = os.environ.get("OUT_DIR", "/outputs")
TAG = os.environ.get("TAG", CONFIG_NAME)
SEED = int(os.environ.get("SEED") or "0")
FPS = int(os.environ.get("FPS") or "6")


def _compose_cfg():
    from omegaconf import OmegaConf
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    for name, fn in (("eval", eval), ("max", lambda x: max(x)),
                     ("split", lambda s, idx: s.split("/")[int(idx)])):
        try:
            OmegaConf.register_new_resolver(name, fn, replace=True)
        except Exception:
            pass
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=os.path.join(AHAWAM_REPO, "configs"), version_base="1.3"):
        return compose(config_name=CONFIG_NAME, overrides=[f"ckpt={CKPT}"])


def _build_dataset(cfg):
    from hydra.utils import instantiate
    import ahawam.datasets.lerobot.robot_video_dataset as rvd
    from ahawam.utils import misc

    def _stub_text_context(self, prompt):
        return torch.zeros(self.context_len, 8), torch.ones(self.context_len, dtype=torch.bool)
    rvd.RobotVideoDataset._get_cached_text_context = _stub_text_context
    try:
        misc.get_work_dir = lambda *a, **k: "/tmp"
    except Exception:
        pass

    return instantiate(
        cfg.data.train,
        dataset_dirs=[DATASET_DIR],
        is_training_set=False,
        val_set_proportion=0.0,
        pretrained_norm_stats=DATASET_STATS,
        skip_padding_as_possible=False,
    )


def _video_tensor_to_frames(video):
    """[C, T, H, W] in [-1,1] -> list of uint8 HxWx3 numpy frames."""
    v = video.detach().float().clamp(-1, 1)
    v = ((v + 1.0) * 127.5).to(torch.uint8).cpu().numpy()
    return [np.ascontiguousarray(v[:, t].transpose(1, 2, 0)) for t in range(v.shape[1])]


def _to_rgb(frame):
    if isinstance(frame, Image.Image):
        return np.array(frame.convert("RGB"))
    return np.asarray(frame)[..., :3]


def _label(img, text):
    pil = Image.fromarray(img.astype(np.uint8))
    ImageDraw.Draw(pil).text((6, 6), text, fill=(255, 255, 0))
    return np.array(pil)


@torch.no_grad()
def _imagine_video(model, input_image, prompt, num_video_frames, num_steps, seed):
    """Denoise the future clip with AHA-WAM's Wan2.2 video expert (first frame pinned)."""
    height, width = int(input_image.shape[-2]), int(input_image.shape[-1])
    context, context_mask = model.encode_prompt(prompt)
    first_frame_latents = model._encode_input_image_latents_tensor(input_image=input_image, tiled=False)

    z_dim = int(model.vae.model.z_dim)
    latent_t = (num_video_frames - 1) // int(model.vae.temporal_downsample_factor) + 1
    latent_h = height // int(model.vae.upsampling_factor)
    latent_w = width // int(model.vae.upsampling_factor)

    generator = torch.Generator(device="cpu").manual_seed(seed)
    latents_video = torch.randn(
        (1, z_dim, latent_t, latent_h, latent_w),
        generator=generator, device="cpu", dtype=torch.float32,
    ).to(device=model.device, dtype=model.torch_dtype)
    latents_video[:, :, 0:1] = first_frame_latents.clone()

    fuse_flag = bool(getattr(model.video_expert, "fuse_vae_embedding_in_latents", False))
    timesteps, deltas = model.infer_video_scheduler.build_inference_schedule(
        num_inference_steps=num_steps, device=model.device,
        dtype=latents_video.dtype, shift_override=None,
    )
    for step_t, step_delta in zip(timesteps, deltas):
        timestep_video = step_t.unsqueeze(0).to(dtype=latents_video.dtype, device=model.device)
        pred_video = model.video_expert(
            x=latents_video, timestep=timestep_video,
            context=context, context_mask=context_mask,
            action=None, fuse_vae_embedding_in_latents=fuse_flag,
        )
        latents_video = model.infer_video_scheduler.step(pred_video, step_delta, latents_video)
        latents_video[:, :, 0:1] = first_frame_latents.clone()

    return model._decode_latents(latents_video, tiled=False)


def main() -> int:
    print(f"torch {torch.__version__}  device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    print(f"config={CONFIG_NAME}  ckpt={os.path.basename(CKPT)}  videos={NUM_VIDEOS}  steps={NUM_STEPS}")

    if AHAWAM_REPO not in sys.path:
        sys.path.insert(0, AHAWAM_REPO)
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    cfg = _compose_cfg()

    t0 = time.time()
    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
    model_cfg.load_text_encoder = True
    model = instantiate(model_cfg, model_dtype=torch.bfloat16, device="cuda")
    model.load_checkpoint(str(CKPT))
    model = model.to("cuda").eval()
    print(f"model loaded: {time.time()-t0:.1f}s  proprio_dim={model.proprio_dim}")

    ds = _build_dataset(cfg)
    starts = ds.lerobot_dataset.episode_data_index["from"].tolist()
    n = min(NUM_VIDEOS, len(starts))
    out_dir = os.path.join(OUT_DIR, f"videogen_{TAG}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"episodes={len(starts)}  generating {n} imagined clips -> {out_dir}\n")

    per = []
    for k in range(n):
        idx = int(starts[k])
        sample = ds[idx]
        video = sample["video"]
        num_video_frames = int(video.shape[1])
        input_image = video[:, 0].unsqueeze(0).to("cuda", dtype=model.torch_dtype)
        prompt = sample["prompt"]

        t1 = time.time()
        gen = _imagine_video(model, input_image, prompt, num_video_frames, NUM_STEPS, SEED)
        latency = time.time() - t1

        gen_frames = [_to_rgb(f) for f in gen]
        gt_frames = _video_tensor_to_frames(video)
        tt = min(len(gen_frames), len(gt_frames))

        stitched = []
        for gt, g in zip(gt_frames[:tt], gen_frames[:tt]):
            if gt.shape[:2] != g.shape[:2]:
                gt = np.array(Image.fromarray(gt).resize((g.shape[1], g.shape[0]), Image.BILINEAR))
            left = _label(gt, "GT")
            right = _label(g, "AHA-WAM imagined")
            stitched.append(np.concatenate([left, right], axis=1))

        mp4 = os.path.join(out_dir, f"clip{k:02d}_gt_vs_imagined.mp4")
        imageio.mimwrite(mp4, stitched, fps=FPS, quality=8, macro_block_size=1)
        per.append({"clip": k, "frame_idx": idx, "frames": tt,
                    "video_latency_s": round(latency, 3), "prompt": prompt[:80]})
        print(f"clip{k:02d} idx={idx:7d} frames={tt} video_latency={latency:.2f}s -> {os.path.basename(mp4)}")

    lat = np.array([p["video_latency_s"] for p in per])
    steady = lat[1:] if len(lat) > 1 else lat
    summary = {
        "tag": TAG, "config": CONFIG_NAME, "num_videos": n,
        "num_inference_steps": NUM_STEPS, "fps": FPS,
        "video_latency_s_mean_all": round(float(lat.mean()), 3),
        "video_latency_s_mean_steady": round(float(steady.mean()), 3),
        "video_latency_s_first_warmup": round(float(lat[0]), 3),
        "clips": per,
    }
    with open(os.path.join(out_dir, "video_latency.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n== {TAG} imagined-video generation ==")
    print(f"video-branch latency ({NUM_STEPS} steps): warmup={lat[0]:.2f}s  steady-mean={steady.mean():.2f}s")
    print(f"videos + video_latency.json -> {out_dir}")
    print("PASS: video generation complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
