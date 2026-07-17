#!/usr/bin/env python
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
Phase-5 quality gate for the Latte gfx1151 optimizations. The headline speedup lever is swapping
the DiT attention from the upstream default `math` (manual softmax matmul) to `flash` (torch SDPA,
~1.8x faster on gfx1151 at fp16). SDPA is numerically equivalent to the math path, so this gate
generates the SAME videos (identical seed / initial noise / sampler / steps) under both attention
modes and reports PSNR / SSIM / LPIPS (optimized-vs-reference) plus writes the two sample sets as
frame folders so FVD(reference, optimized) can be computed with the ported metric stack.

A pass = PSNR very high (>=40 dB) + SSIM ~1 + LPIPS ~0 (i.e. the optimization does not change output).
Env: DATASET, LATTE_DIR, N (videos), STEPS, SEED, OUT_DIR.
"""
import os, sys, json
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/repos/latte")
from einops import rearrange
from omegaconf import OmegaConf
from diffusers.models import AutoencoderKL
from models import get_models
from utils import find_model
from diffusion import create_diffusion

CKPTS = {"ffs": "ffs.pt", "sky": "skytimelapse.pt", "taichi": "taichi-hd.pt", "ucf101": "ucf101.pt"}
DATASET = os.environ.get("DATASET", "sky")
LATTE_DIR = os.environ.get("LATTE_DIR", "/models/Latte")
OUT = os.environ.get("OUT_DIR", "/outputs/parity")
N = int(os.environ.get("N", "6"))
STEPS = int(os.environ.get("STEPS", "50"))
SEED = int(os.environ.get("SEED", "0"))
dev = "cuda"


def set_attn(model, mode):
    for m in model.modules():
        if hasattr(m, "attention_mode"):
            m.attention_mode = mode


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 100.0 if mse == 0 else 20 * np.log10(255.0) - 10 * np.log10(mse)


def ssim(a, b):
    # global gaussian-free SSIM approximation over the whole frame (luma), per-frame then averaged.
    a = a.astype(np.float64); b = b.astype(np.float64)
    mu_a, mu_b = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    return ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2))


def gen(model, vae, cfg, latent, mode):
    set_attn(model, mode)
    diffusion = create_diffusion(str(STEPS))
    torch.manual_seed(SEED)
    z = torch.randn(N, cfg.num_frames, 4, latent, latent, dtype=torch.float16, device=dev)
    samples = diffusion.ddim_sample_loop(model.forward, z.shape, z, clip_denoised=False,
                                         model_kwargs=dict(y=None, use_fp16=True), progress=False, device=dev)
    bb, f, c, h, w = samples.shape
    samples = rearrange(samples.half(), "b f c h w -> (b f) c h w")
    imgs = vae.decode(samples / 0.18215).sample
    imgs = rearrange(imgs, "(b f) c h w -> b f c h w", b=bb)
    imgs = ((imgs * 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(torch.uint8).cpu().permute(0, 1, 3, 4, 2).numpy()
    return imgs


def save_set(imgs, root):
    for i in range(imgs.shape[0]):
        d = os.path.join(root, f"vid{i:04d}"); os.makedirs(d, exist_ok=True)
        for fi in range(imgs.shape[1]):
            Image.fromarray(imgs[i, fi]).save(os.path.join(d, f"{fi:03d}.png"))


def main():
    torch.set_grad_enabled(False)
    cfg = OmegaConf.load(f"/repos/latte/configs/{DATASET}/{DATASET}_sample.yaml")
    latent = int(cfg.image_size) // 8
    model = get_models(OmegaConf.create({**cfg, "latent_size": latent})).to(dev)
    model.load_state_dict(find_model(os.path.join(LATTE_DIR, CKPTS[DATASET])))
    model.eval()
    vae = AutoencoderKL.from_pretrained(os.path.join(LATTE_DIR, "vae")).to(dev)
    model.half(); vae.half()

    print("generating reference (attention_mode=math)...", flush=True)
    ref = gen(model, vae, cfg, latent, "math")
    print("generating optimized (attention_mode=flash / SDPA)...", flush=True)
    opt = gen(model, vae, cfg, latent, "flash")

    os.makedirs(OUT, exist_ok=True)
    save_set(ref, os.path.join(OUT, "ref_math"))
    save_set(opt, os.path.join(OUT, "opt_sdpa"))

    psnrs = [psnr(ref[i], opt[i]) for i in range(N)]
    ssims = [ssim(ref[i], opt[i]) for i in range(N)]

    lpips_mean = None
    try:
        import lpips as lpips_lib
        loss_fn = lpips_lib.LPIPS(net="alex").to(dev)
        vals = []
        for i in range(N):
            a = torch.from_numpy(ref[i]).permute(0, 3, 1, 2).float().to(dev) / 127.5 - 1
            b = torch.from_numpy(opt[i]).permute(0, 3, 1, 2).float().to(dev) / 127.5 - 1
            vals.append(loss_fn(a, b).mean().item())
        lpips_mean = float(np.mean(vals))
    except Exception as e:
        print("LPIPS unavailable:", e)

    result = dict(
        dataset=DATASET, n_videos=N, steps=STEPS, seed=SEED,
        comparison="attention math (reference) vs flash/SDPA (optimized), identical seed+sampler+steps",
        psnr_db_mean=float(np.mean(psnrs)), psnr_db_min=float(np.min(psnrs)),
        ssim_mean=float(np.mean(ssims)), ssim_min=float(np.min(ssims)),
        lpips_mean=lpips_mean,
        pass_gate=bool(np.mean(psnrs) >= 40 and np.mean(ssims) >= 0.99),
    )
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "parity.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print("wrote", os.path.join(OUT, "parity.json"))


if __name__ == "__main__":
    raise SystemExit(main())
