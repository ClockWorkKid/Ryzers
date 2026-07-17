#!/usr/bin/env python
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
B2 validation: fp32 vs bf16 VAE DECODE quality + speed. Encodes real csgo frames (fp32), then decodes
the latents in fp32 and bf16, reporting per-frame PSNR/max-abs and the decode wall-clock speedup.
Saves a fp32|bf16|10x|abs-diff montage so we can eyeball the known bf16 'colored speckle' failure mode.
"""
import os, sys, time, argparse
import numpy as np
import torch
import imageio.v2 as imageio
from omegaconf import OmegaConf
from PIL import Image

sys.path.insert(0, "/repos/nanowm"); sys.path.insert(0, "/repos/nanowm/src")
from latent_codecs import load_autoencoder_kl, resolve_latent_codec_config
from sample.sampling_utils import encode_frames, decode_latents


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="/models/results/csgo/config.yaml")
    ap.add_argument("--video", default="/outputs/csgo_multi/seed0/sample_0000_gt.mp4")
    ap.add_argument("--vae_model_path", default="stabilityai/sd-vae-ft-mse")
    ap.add_argument("--out", default="/outputs/video3d/vae_bf16_check.png")
    ap.add_argument("--nframes", type=int, default=16)
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config); OmegaConf.set_struct(cfg, False)
    dev = "cuda"
    codec = resolve_latent_codec_config(cfg)
    vae = load_autoencoder_kl(args.vae_model_path or codec.model_path).to(dev).eval()

    r = imageio.get_reader(args.video)
    frames = [f[..., :3] for f in r]; r.close()
    frames = np.stack(frames[:args.nframes]).astype(np.float32) / 255.0
    x = torch.from_numpy(frames).permute(0, 3, 1, 2).unsqueeze(0).to(dev)  # [1,F,3,H,W]
    x = x * 2.0 - 1.0

    with torch.no_grad():
        lat = encode_frames(vae, x, dev, vae_precision="fp32")

        torch.cuda.synchronize(); t0 = time.time()
        f32 = decode_latents(vae, lat, vae_precision="fp32"); torch.cuda.synchronize()
        t_f32 = time.time() - t0

        torch.cuda.synchronize(); t0 = time.time()
        bf16 = decode_latents(vae, lat, vae_precision="bf16"); torch.cuda.synchronize()
        t_bf16 = time.time() - t0

    f32c = f32[0].float().clamp(0, 1); bf16c = bf16[0].float().clamp(0, 1)
    diff = (f32c - bf16c).abs()
    mse = diff.pow(2).mean().item()
    psnr = 10 * np.log10(1.0 / max(mse, 1e-12))
    print(f"[B2] frames={f32c.shape[0]} res={tuple(f32c.shape[-2:])}")
    print(f"[B2] decode fp32={t_f32*1000:.0f}ms  bf16={t_bf16*1000:.0f}ms  speedup={t_f32/max(t_bf16,1e-6):.2f}x")
    print(f"[B2] bf16-vs-fp32 decode: PSNR={psnr:.2f} dB  MSE={mse:.2e}  max_abs={diff.max().item():.3e}")

    # montage for 2 frames: fp32 | bf16 | 10x abs-diff
    def row(i):
        a = (f32c[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        b = (bf16c[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        d = (np.clip(diff[i].permute(1, 2, 0).cpu().numpy() * 10, 0, 1) * 255).astype(np.uint8)
        sep = np.full((a.shape[0], 3, 3), 255, np.uint8)
        return np.concatenate([a, sep, b, sep, d], axis=1)
    idxs = [min(f32c.shape[0] - 1, k) for k in (0, f32c.shape[0] // 2, f32c.shape[0] - 1)]
    grid = np.concatenate([row(i) for i in idxs], axis=0)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    Image.fromarray(grid).save(args.out)
    print(f"[B2] wrote {args.out}  (columns: fp32 | bf16 | 10x|diff|)  ({os.path.getsize(args.out)//1024} KB)")


if __name__ == "__main__":
    raise SystemExit(main())
