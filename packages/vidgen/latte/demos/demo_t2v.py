# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Latte text-to-video / text-to-image demo entrypoint (Strix Halo, gfx1151, ROCm).

A thin CLI on top of the upstream APIs (`sample/pipeline_latte.LattePipeline`,
`models.get_models`, diffusers schedulers) - it does NOT reimplement the model, it
just exposes every generation knob the upstream pipeline already supports so you can
probe the model's capability before committing:

  * --steps        num_sampling_steps (diffusion denoising steps)
  * --resolution   HxW image size (must be divisible by 8)
  * --frames       video_length (1 => text-to-image, >1 => text-to-video)
  * --guidance     classifier-free guidance scale
  * --method       sampler: DDIM/DDPM/PNDM/EulerDiscrete/EulerAncestralDiscrete/
                   DPMSolverMultistep/DPMSolverSinglestep/HeunDiscrete/DEISMultistep/
                   KDPM2AncestralDiscrete
  * --seed         reproducible seed (wired via torch.Generator; upstream leaves it off)
  * --fps          output mp4 frame rate (upstream hardcodes 8)
  * --temporal-vae / --no-temporal-vae   SVD temporal VAE decoder (less flicker)
  * --temporal-attn / --no-temporal-attn temporal attention blocks
  * --fp16 / --fp32                       transformer/VAE/T5 compute precision

Weights: HF `maxin-cn/Latte-1` (default /models/Latte-1), fetched by
`scripts/download_checkpoints.sh WHICH=t2v`.
"""
import argparse
import os
import sys
import time

import torch

# Upstream layout: repo root on path + the sample/ dir (pipeline_latte lives there).
_REPO = os.environ.get("LATTE_REPO", "/repos/latte")
for _p in (_REPO, os.path.join(_REPO, "sample")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from diffusers.models import AutoencoderKL, AutoencoderKLTemporalDecoder  # noqa: E402
from diffusers.schedulers import (  # noqa: E402
    DDIMScheduler, DDPMScheduler, PNDMScheduler, EulerDiscreteScheduler,
    DPMSolverMultistepScheduler, HeunDiscreteScheduler,
    EulerAncestralDiscreteScheduler, DEISMultistepScheduler,
    KDPM2AncestralDiscreteScheduler,
)
from diffusers.schedulers.scheduling_dpmsolver_singlestep import (  # noqa: E402
    DPMSolverSinglestepScheduler,
)
from transformers import T5EncoderModel, T5Tokenizer  # noqa: E402
from torchvision.utils import save_image  # noqa: E402
import imageio  # noqa: E402

from models import get_models  # noqa: E402  (upstream)
from pipeline_latte import LattePipeline  # noqa: E402  (upstream, in sample/)

# Sampler name -> (class, extra kwargs, pass_variance_type).
# Only DDIM/DDPM consume `variance_type`; the DPM/Euler/etc. family mis-slices the
# 4-channel Latte latent to 3 channels when `variance_type='learned_range'` is passed
# (it assumes RGB). The pipeline already strips the learned variance before scheduler.step,
# so we simply omit variance_type for those. (Upstream passes it to all -> those samplers
# are broken upstream for T2V; here they work.)
_SCHEDULERS = {
    "DDIM": (DDIMScheduler, {"clip_sample": False}, True),
    "DDPM": (DDPMScheduler, {"clip_sample": False}, True),
    "PNDM": (PNDMScheduler, {}, False),
    "EulerDiscrete": (EulerDiscreteScheduler, {}, False),
    "EulerAncestralDiscrete": (EulerAncestralDiscreteScheduler, {}, False),
    "DPMSolverMultistep": (DPMSolverMultistepScheduler, {}, False),
    "DPMSolverSinglestep": (DPMSolverSinglestepScheduler, {}, False),
    "HeunDiscrete": (HeunDiscreteScheduler, {}, False),
    "DEISMultistep": (DEISMultistepScheduler, {}, False),
    "KDPM2AncestralDiscrete": (KDPM2AncestralDiscreteScheduler, {}, False),
}


def build_scheduler(method, model_path, beta_start, beta_end, beta_schedule, variance_type):
    if method not in _SCHEDULERS:
        raise SystemExit(f"unknown --method {method}; choose from {list(_SCHEDULERS)}")
    cls, extra, pass_variance = _SCHEDULERS[method]
    kwargs = dict(beta_start=beta_start, beta_end=beta_end, beta_schedule=beta_schedule, **extra)
    if pass_variance:
        kwargs["variance_type"] = variance_type
    return cls.from_pretrained(model_path, subfolder="scheduler", **kwargs)


def parse_args():
    p = argparse.ArgumentParser(
        description="Latte text-to-video / text-to-image demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--prompt", action="append", default=None,
                   help="text prompt; repeat --prompt for a batch (falls back to a demo prompt)")
    p.add_argument("--steps", type=int, default=50, help="diffusion sampling steps")
    p.add_argument("--resolution", type=str, default="512x512", help="HxW, each divisible by 8")
    p.add_argument("--frames", type=int, default=16, help="video_length; 1 => text-to-image")
    p.add_argument("--guidance", type=float, default=7.5, help="classifier-free guidance scale")
    p.add_argument("--method", type=str, default="DDIM", help="sampler (see --help)")
    p.add_argument("--seed", type=int, default=0, help="random seed (reproducible)")
    p.add_argument("--fps", type=int, default=8, help="output mp4 frame rate")
    p.add_argument("--num-per-prompt", type=int, default=1, help="samples per prompt")
    p.add_argument("--temporal-vae", dest="temporal_vae", action="store_true", default=None,
                   help="use SVD temporal VAE decoder (default: on for video, off for image)")
    p.add_argument("--no-temporal-vae", dest="temporal_vae", action="store_false")
    p.add_argument("--temporal-attn", dest="temporal_attn", action="store_true", default=True)
    p.add_argument("--no-temporal-attn", dest="temporal_attn", action="store_false")
    p.add_argument("--fp32", dest="fp16", action="store_false", default=True,
                   help="run in fp32 (default fp16)")
    p.add_argument("--model", type=str, default=os.environ.get("LATTE1_DIR", "/models/Latte-1"))
    p.add_argument("--out", type=str, default=os.environ.get("OUT_DIR", "/outputs") + "/t2v_demo")
    # scheduler beta schedule (upstream defaults)
    p.add_argument("--beta-start", type=float, default=0.0001)
    p.add_argument("--beta-end", type=float, default=0.02)
    p.add_argument("--beta-schedule", type=str, default="linear")
    p.add_argument("--variance-type", type=str, default="learned_range")
    return p.parse_args()


def main():
    args = parse_args()
    try:
        height, width = (int(x) for x in args.resolution.lower().split("x"))
    except ValueError:
        raise SystemExit(f"--resolution must be HxW (got {args.resolution!r})")
    if height % 8 or width % 8:
        raise SystemExit(f"--resolution {height}x{width}: H and W must be divisible by 8")

    prompts = args.prompt or ["a cat wearing sunglasses and working as a lifeguard at pool"]
    temporal_vae = args.temporal_vae
    if temporal_vae is None:
        temporal_vae = args.frames > 1  # video benefits from the temporal decoder

    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if args.fp16 else torch.float32

    print("=" * 68)
    print("Latte T2V demo | device:", device, "| dtype:", dtype)
    print(f"  prompts={len(prompts)} steps={args.steps} res={height}x{width} "
          f"frames={args.frames} guidance={args.guidance} method={args.method}")
    print(f"  seed={args.seed} fps={args.fps} temporal_vae={temporal_vae} "
          f"temporal_attn={args.temporal_attn} num_per_prompt={args.num_per_prompt}")
    print(f"  model={args.model}")
    print("=" * 68, flush=True)

    # --- load (upstream get_models + T5 + SD/temporal VAE) ---
    t0 = time.time()
    model_args = argparse.Namespace(model="LatteT2V", pretrained_model_path=args.model,
                                    video_length=args.frames)
    transformer = get_models(model_args).to(device, dtype=dtype).eval()
    if temporal_vae:
        vae = AutoencoderKLTemporalDecoder.from_pretrained(
            args.model, subfolder="vae_temporal_decoder", torch_dtype=dtype).to(device)
    else:
        vae = AutoencoderKL.from_pretrained(
            args.model, subfolder="vae", torch_dtype=dtype).to(device)
    tokenizer = T5Tokenizer.from_pretrained(args.model, subfolder="tokenizer")
    text_encoder = T5EncoderModel.from_pretrained(
        args.model, subfolder="text_encoder", torch_dtype=dtype).to(device).eval()
    scheduler = build_scheduler(args.method, args.model, args.beta_start, args.beta_end,
                                args.beta_schedule, args.variance_type)
    pipe = LattePipeline(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
                         scheduler=scheduler, transformer=transformer).to(device)
    print(f"[load] models ready in {time.time() - t0:.1f}s", flush=True)

    os.makedirs(args.out, exist_ok=True)
    saved = []
    for i, prompt in enumerate(prompts):
        gen = torch.Generator(device=device).manual_seed(args.seed + i)
        print(f"[gen {i + 1}/{len(prompts)}] {prompt!r}", flush=True)
        t1 = time.time()
        videos = pipe(
            prompt,
            video_length=args.frames,
            height=height, width=width,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            num_images_per_prompt=args.num_per_prompt,
            generator=gen,
            mask_feature=True,
            enable_temporal_attentions=args.temporal_attn,
            enable_vae_temporal_decoder=temporal_vae,
        ).video
        dt = time.time() - t1
        stem = os.path.join(args.out, f"{i:02d}_" + "_".join(prompt.split())[:60])
        if videos.shape[1] == 1:  # single frame -> image (decode_latents_image: CHW float[0,1])
            path = stem + ".png"
            save_image(videos[0][0], path)
        else:  # video (decode_latents: FHWC uint8)
            path = stem + ".mp4"
            imageio.mimwrite(path, videos[0], fps=args.fps, quality=9)
        n = videos.shape[1]
        print(f"[gen {i + 1}/{len(prompts)}] {n} frame(s) in {dt:.1f}s "
              f"({dt / max(args.steps, 1):.2f}s/step) -> {path}", flush=True)
        saved.append(path)

    print("=" * 68)
    print("[done] wrote", len(saved), "output(s):")
    for s in saved:
        print("  -", s)
    print("=" * 68, flush=True)


if __name__ == "__main__":
    main()
