# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Unified, env-driven Micro-World inference runner for all four upstream examples, without editing
the upstream scripts in place (the demos/*.sh wrap this). Reproduces the exact model-loading +
sampling of examples/wan2.1/predict_{t2v,t2w_action_control,i2v,i2w_action_control}.py, selected by
--mode, with every knob overridable from the environment (see config.yaml / the demo .sh headers).

Output (rule 2.b):
  * t2v / t2w  — pure generative, no reference -> single-column MP4.
  * i2v / i2w  — image-to-video -> two-column MP4: reference image (left) | generated video (right).

Run from the repo root (/repos/micro-world) so the config's relative subpaths resolve.
"""
import argparse
import json
import os
from datetime import datetime

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from PIL import Image
from transformers import AutoTokenizer

from microworld.models import (AutoencoderKLWan, CLIPModel, WanT5EncoderModel,
                               WanTransformer3DModel, WanActionControlNetModel, WanActionAdaLNModel)
from microworld.models.cache_utils import get_teacache_coefficients
from microworld.pipeline import (WanFunPipeline, WanFunInpaintPipeline,
                                 WanActionT2WPipeline, WanActionI2WPipeline)
from microworld.utils.lora_utils import merge_lora, unmerge_lora
from microworld.utils.utils import (filter_kwargs, replace_parameters_by_name, save_videos_grid,
                                     parse_action_list, get_image_to_video_latent)
from microworld.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from microworld.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

DEVICE = "cuda"
DTYPE = torch.bfloat16
MODELS_DIR = os.environ.get("MODELS_DIR", "/models")


def apply_gfx1151_speedups():
    """Kernel-level gfx1151 speedup: conv3d cudnn override (default ON, kill-switch MW_DISABLE_CUDNN=0).

    Micro-World's AutoencoderKLWan is the same conv3d WAN VAE that dominated VERA. gfx1151 MIOpen has
    no tuned 3D conv solver for these shapes -> a naive fallback that dominates VAE encode/decode (and
    pays a large one-time autotune). `torch.backends.cudnn.enabled=False` routes every conv through
    ATen im2col/unfold + GEMM (rocBLAS/hipBLASLt) -- exact (no algorithm change), measured ~8.8x on
    the WAN conv3d in the VERA port. The DiT/T5 already run bf16 with flash SDPA + built-in autocast,
    so this is the single portable lever that is not already captured by the pipeline.
    """
    if os.environ.get("MW_DISABLE_CUDNN", "1") == "1":
        torch.backends.cudnn.enabled = False
        print("[gfx1151-speedup] cudnn disabled (conv3d via ATen im2col+GEMM)")


apply_gfx1151_speedups()

# Default English negative prompt (upstream uses a longer Chinese one; set NEGATIVE_PROMPT to match
# the paper settings for best fidelity). Kept ASCII-safe for robust cross-platform transfer.
DEFAULT_NEG = "bad detailed, static, blur, messy, error, distorted, low quality, watermark, text"

# Per-mode defaults mirroring the upstream example scripts.
MODES = {
    "t2v": dict(
        base="Diffusion_Transformer/Wan2.1-T2V-1.3B", pipeline="fun", clip=False, action=False,
        guidance=3.0, fps=15, teacache_threshold=0.10, transformer=None, lora=None, train_mode=None,
        prompt=("A first-person view walking through a blocky Minecraft world: a sandy desert with "
                "cacti and scattered stone platforms under a clear sky, part of a sword visible in "
                "the foreground."),
        ref=None, action_list=None),
    "t2w": dict(
        base="Diffusion_Transformer/Wan2.1-T2V-1.3B", pipeline="t2w", clip=False, action=True,
        guidance=3.0, fps=15, teacache_threshold=0.20, train_mode="controlnet",
        transformer=os.path.join(MODELS_DIR, "T2W/transformer"), lora=None,
        prompt=("Running along a cliffside path on a tropical island in first person perspective, "
                "turquoise waters crashing against the rocks far below, the path twisting along the "
                "jagged cliffs."),
        ref=None, action_list=[[20, "1 0 0 0 0 0 0 0 0"], [40, "0 1 0 0 0 0 0 0 5"],
                               [80, "0 0 0 1 0 0 0 0 -3"], "30 60"]),
    "i2v": dict(
        base="Diffusion_Transformer/Wan2.1-I2V-14B-480P", pipeline="fun_inpaint", clip=True,
        action=False, guidance=6.0, fps=16, teacache_threshold=0.10, transformer=None, lora=None,
        train_mode=None,
        prompt=("Running along a cliffside path on a tropical island in first person perspective, "
                "turquoise waters crashing against the rocks far below."),
        ref="asset/cliff.jpg", action_list=None),
    "i2w": dict(
        base="Diffusion_Transformer/Wan2.1-I2V-14B-480P", pipeline="i2w", clip=True, action=True,
        guidance=6.0, fps=16, teacache_threshold=0.20, train_mode="adaln",
        transformer=os.path.join(MODELS_DIR, "I2W/transformer"),
        lora=os.path.join(MODELS_DIR, "I2W/lora_diffusion_pytorch_model.safetensors"),
        prompt=("First-person perspective walking down a lively city street at night, neon signs "
                "and bright billboards glowing on both sides, immersive urban night scene."),
        ref="asset/street_night.jpg",
        action_list=[[20, "0 1 0 1 0 0 0 0 0"], [40, "1 0 0 0 0 0 0 0 0"], [80, "0 0 1 0 0 0 0 0 0"], "40"]),
}


def env(name, default=None):
    v = os.environ.get(name)
    return default if v is None or v == "" else v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=list(MODES))
    args = ap.parse_args()
    d = MODES[args.mode]
    mode = args.mode

    cfg_path = env("CONFIG", "/repos/micro-world/config/wan2.1/wan_civitai.yaml")
    config = OmegaConf.load(cfg_path)

    base_model = env("BASE_MODEL", os.path.join(MODELS_DIR, d["base"]))
    transformer_path = env("TRANSFORMER_PATH", d["transformer"])
    lora_path = env("LORA_PATH", d["lora"])
    lora_weight = float(env("LORA_WEIGHT", "1.0"))
    train_mode = env("TRAIN_MODE", d["train_mode"])

    prompt = env("PROMPT", d["prompt"])
    negative_prompt = env("NEGATIVE_PROMPT", DEFAULT_NEG)
    # 14B I2V/I2W transformers exceed the 47GB iGPU when resident -> default them to cpu offload.
    default_gpu_mode = "model_cpu_offload" if d["base"].endswith("I2V-14B-480P") else "model_full_load"
    gpu_memory_mode = env("GPU_MEMORY_MODE", default_gpu_mode)
    sampler_name = env("SAMPLER", "Flow_Unipc")
    shift = float(env("SHIFT", "3"))
    ss = env("SAMPLE_SIZE", "352,640")
    sample_size = [int(x) for x in ss.replace("x", ",").split(",")]
    video_length = int(env("VIDEO_LENGTH", "81"))
    fps = int(env("FPS", str(d["fps"])))
    guidance_scale = float(env("GUIDANCE_SCALE", str(d["guidance"])))
    seed = int(env("SEED", "43"))
    num_inference_steps = int(env("NUM_STEPS", "30"))
    enable_teacache = env("TEACACHE", "1") not in ("0", "false", "False")
    teacache_threshold = float(env("TEACACHE_THRESHOLD", str(d["teacache_threshold"])))
    num_skip_start_steps = int(env("NUM_SKIP_START_STEPS", "5"))
    teacache_offload = env("TEACACHE_OFFLOAD", "0") in ("1", "true", "True")
    cfg_skip_ratio = float(env("CFG_SKIP_RATIO", "0"))
    ref_image = env("REF_IMAGE", d["ref"])
    out_dir = env("OUT_DIR", "/outputs")

    action_list = d["action_list"]
    if d["action"] and env("ACTION_LIST"):
        action_list = json.loads(env("ACTION_LIST"))

    # Effective frame count: the clip (i2v/i2w) paths round video_length to the VAE temporal grid
    # (4n+1). Compute it up front (tcr from config) so we can clamp the action schedule to it.
    tcr = int(config["vae_kwargs"].get("temporal_compression_ratio", 4))
    if d["clip"]:
        video_length = int((video_length - 1) // tcr * tcr) + 1 if video_length != 1 else 1

    def _clamp_actions(al, maxf):
        # Clamp action end-frames (and the trailing "space frames" string) to maxf so shorter
        # validation runs still work with the default schedule (no fragile env override needed).
        out = []
        for e in al:
            if isinstance(e, (list, tuple)):
                out.append([min(int(e[0]), maxf), e[1]])
            else:
                out.append(" ".join(str(min(int(x), maxf)) for x in str(e).split()))
        return out

    if d["action"]:
        action_list = _clamp_actions(action_list, video_length - 1)

    print(f"=== Micro-World {mode} ===")
    print(f"base={base_model}\ntransformer={transformer_path}\nlora={lora_path}\n"
          f"sample_size={sample_size} frames={video_length} steps={num_inference_steps} "
          f"guidance={guidance_scale} sampler={sampler_name} shift={shift} seed={seed}\n"
          f"gpu_memory_mode={gpu_memory_mode} teacache={enable_teacache}({teacache_threshold}) "
          f"cfg_skip={cfg_skip_ratio}")
    if d["action"]:
        print(f"action_list={action_list}")

    tkw = OmegaConf.to_container(config["transformer_additional_kwargs"])

    # --- Transformer / action model ---
    if d["pipeline"] in ("fun", "fun_inpaint"):
        tf_src = transformer_path if transformer_path else os.path.join(base_model, tkw.get("transformer_subpath", "./"))
        transformer = WanTransformer3DModel.from_pretrained(
            tf_src, transformer_additional_kwargs=tkw, low_cpu_mem_usage=True, torch_dtype=DTYPE)
    else:
        model_class = WanActionControlNetModel if train_mode == "controlnet" else WanActionAdaLNModel
        transformer = model_class.from_pretrained(
            transformer_path, transformer_additional_kwargs=tkw, low_cpu_mem_usage=True, torch_dtype=DTYPE)

    # --- VAE (conv3d) ---
    vae = AutoencoderKLWan.from_pretrained(
        os.path.join(base_model, config["vae_kwargs"].get("vae_subpath", "Wan2.1_VAE.pth")),
        additional_kwargs=OmegaConf.to_container(config["vae_kwargs"]),
    ).to(DTYPE)

    # --- Tokenizer + T5 text encoder ---
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(base_model, config["text_encoder_kwargs"].get("tokenizer_subpath")))
    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(base_model, config["text_encoder_kwargs"].get("text_encoder_subpath")),
        additional_kwargs=OmegaConf.to_container(config["text_encoder_kwargs"]),
        low_cpu_mem_usage=True, torch_dtype=DTYPE).eval()

    # --- CLIP image encoder (i2v/i2w) ---
    clip_image_encoder = None
    if d["clip"]:
        clip_image_encoder = CLIPModel.from_pretrained(
            os.path.join(base_model, config["image_encoder_kwargs"].get("image_encoder_subpath")),
        ).to(DTYPE).eval()

    # --- Scheduler ---
    Scheduler = {"Flow": FlowMatchEulerDiscreteScheduler,
                 "Flow_Unipc": FlowUniPCMultistepScheduler,
                 "Flow_DPM++": FlowDPMSolverMultistepScheduler}[sampler_name]
    if sampler_name in ("Flow_Unipc", "Flow_DPM++"):
        config["scheduler_kwargs"]["shift"] = 1
    scheduler = Scheduler(**filter_kwargs(Scheduler, OmegaConf.to_container(config["scheduler_kwargs"])))

    # --- Pipeline ---
    pk = dict(transformer=transformer, vae=vae, tokenizer=tokenizer,
              text_encoder=text_encoder, scheduler=scheduler)
    if d["pipeline"] == "fun":
        pipeline = WanFunPipeline(**pk)
    elif d["pipeline"] == "t2w":
        pipeline = WanActionT2WPipeline(**pk)
    elif d["pipeline"] == "fun_inpaint":
        pipeline = WanFunInpaintPipeline(clip_image_encoder=clip_image_encoder, **pk)
    else:
        pipeline = WanActionI2WPipeline(clip_image_encoder=clip_image_encoder, **pk)

    if gpu_memory_mode == "sequential_cpu_offload":
        replace_parameters_by_name(transformer, ["modulation"], device=DEVICE)
        transformer.freqs = transformer.freqs.to(device=DEVICE)
        pipeline.enable_sequential_cpu_offload(device=DEVICE)
    elif gpu_memory_mode == "model_cpu_offload":
        pipeline.enable_model_cpu_offload(device=DEVICE)
    else:
        pipeline.to(device=DEVICE)

    coefficients = get_teacache_coefficients(base_model) if enable_teacache else None
    if coefficients is not None:
        print(f"TeaCache: threshold={teacache_threshold} skip_start={num_skip_start_steps}")
        pipeline.transformer.enable_teacache(
            coefficients, num_inference_steps, teacache_threshold,
            num_skip_start_steps=num_skip_start_steps, offload=teacache_offload)
    if cfg_skip_ratio and hasattr(pipeline.transformer, "enable_cfg_skip"):
        print(f"cfg_skip_ratio={cfg_skip_ratio}")
        pipeline.transformer.enable_cfg_skip(cfg_skip_ratio, num_inference_steps)

    generator = torch.Generator(device=DEVICE).manual_seed(seed)

    kb = ms = None
    if d["action"]:
        kb, ms = parse_action_list(action_list)
        kb = kb[None].to(device=DEVICE, dtype=DTYPE)
        ms = ms[None].to(device=DEVICE, dtype=DTYPE)

    if lora_path:
        pipeline = merge_lora(pipeline, lora_path, lora_weight, device=DEVICE)

    input_video = None
    with torch.no_grad():
        common = dict(num_frames=video_length, negative_prompt=negative_prompt,
                      height=sample_size[0], width=sample_size[1], generator=generator,
                      guidance_scale=guidance_scale, num_inference_steps=num_inference_steps, shift=shift)
        if d["action"]:
            common.update(mouse_actions=ms, keyboard_actions=kb)

        if d["clip"]:
            # video_length was already rounded to the VAE temporal grid above.
            input_video, input_video_mask, _ = get_image_to_video_latent(
                ref_image, None, video_length=video_length, sample_size=sample_size)
            input_video_mask = torch.zeros_like(input_video[:, :1])
            input_video_mask[:, :, 1:] = 255
            clip_image = (input_video[0, :, 0] * 255).to(torch.uint8).permute(1, 2, 0)
            clip_image = Image.fromarray(clip_image.cpu().numpy())
            common.update(video=input_video, mask_video=input_video_mask, clip_image=clip_image)

        sample = pipeline(prompt, **common).videos

    if lora_path:
        pipeline = unmerge_lora(pipeline, lora_path, lora_weight, device=DEVICE)

    # rule 2.b: image-to-video -> reference image (left) | generated (right); pure gen -> single.
    if d["clip"] and input_video is not None and video_length != 1:
        ref_col = input_video[:, :, 0:1].to(sample.device, sample.dtype).repeat(1, 1, sample.shape[2], 1, 1)
        grid = torch.cat([ref_col, sample], dim=-1)
    else:
        grid = sample

    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if video_length == 1:
        img = (sample[0, :, 0].transpose(0, 1).transpose(1, 2) * 255).numpy().astype(np.uint8)
        path = os.path.join(out_dir, f"mw_{mode}_{stamp}.png")
        Image.fromarray(img).save(path)
    else:
        path = os.path.join(out_dir, f"mw_{mode}_{stamp}.mp4")
        save_videos_grid(grid, path, fps=fps)
    print(f"SAVED {path}")


if __name__ == "__main__":
    main()
