# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Derive the flux2-native `ae.safetensors` that ImageWAM's FLUX.2 loader expects from the
autoencoder bundled inside the (accessible) FLUX.2-klein-base-4B repo, WITHOUT needing the
separately-gated FLUX.2-dev repo.

Why this exists: ImageWAM loads the AE with `AutoEncoder(AutoEncoderParams());
load_state_dict(load_sft(ae_path), strict=True)`. Upstream's helper downloads that file from
`black-forest-labs/FLUX.2-dev` (a separately-gated repo). But the klein-base-4B repo we DO
have ships the *identical* autoencoder (`AutoencoderKLFlux2`, verified 251/251 same-shape
params) as `vae/diffusion_pytorch_model.safetensors`, only in diffusers key-naming. This
converts that diffusers state dict to the flux2-native `AutoEncoder` layout — a purely
mechanical, well-known key remap (same-order down blocks; reversed up blocks; attention
Linear->1x1 Conv; group_norm->norm; conv_shortcut->nin_shortcut; conv_norm_out->norm_out;
quant_conv->encoder.quant_conv; post_quant_conv->decoder.post_quant_conv).

Correctness is *verified*, not assumed: (1) strict load into the flux2 AutoEncoder (proves
key/shape completeness), (2) encode->decode reconstruction PSNR on a textured image (a wrong
layer assignment collapses this), and (3) if the installed diffusers exposes
`AutoencoderKLFlux2`, a direct numerical cross-check that the converted AE reproduces the
reference model's reconstruction. The ultimate check is P5 open-loop (dream vs GT). If the
dev later obtains FLUX.2-dev access, the official ae.safetensors is a drop-in replacement.

Usage (in the imagewam container, /models mounted):
  python /ryzers/scripts/convert_klein_vae.py \
     --src /models/flux2/FLUX.2-klein-base-4B/vae/diffusion_pytorch_model.safetensors \
     --out /models/flux2/FLUX.2-klein-base-4B/ae.safetensors
"""
import argparse
import os
import sys

import torch
from safetensors.torch import load_file, save_file


def _resnet(dst, dp, src, sp):
    for t in ("norm1", "conv1", "norm2", "conv2"):
        dst[f"{dp}.{t}.weight"] = src[f"{sp}.{t}.weight"]
        dst[f"{dp}.{t}.bias"] = src[f"{sp}.{t}.bias"]
    if f"{sp}.conv_shortcut.weight" in src:  # channel-changing block
        dst[f"{dp}.nin_shortcut.weight"] = src[f"{sp}.conv_shortcut.weight"]
        dst[f"{dp}.nin_shortcut.bias"] = src[f"{sp}.conv_shortcut.bias"]


def _attn(dst, dp, src, sp):
    dst[f"{dp}.norm.weight"] = src[f"{sp}.group_norm.weight"]
    dst[f"{dp}.norm.bias"] = src[f"{sp}.group_norm.bias"]
    for tgt, s in (("q", "to_q"), ("k", "to_k"), ("v", "to_v"), ("proj_out", "to_out.0")):
        w = src[f"{sp}.{s}.weight"]                    # diffusers Linear (C, C)
        dst[f"{dp}.{tgt}.weight"] = w.reshape(w.shape[0], w.shape[1], 1, 1)  # flux 1x1 Conv
        dst[f"{dp}.{tgt}.bias"] = src[f"{sp}.{s}.bias"]


def convert(src):
    dst = {}
    # top-level
    dst["encoder.quant_conv.weight"] = src["quant_conv.weight"]
    dst["encoder.quant_conv.bias"] = src["quant_conv.bias"]
    dst["decoder.post_quant_conv.weight"] = src["post_quant_conv.weight"]
    dst["decoder.post_quant_conv.bias"] = src["post_quant_conv.bias"]
    for t in ("running_mean", "running_var", "num_batches_tracked"):
        dst[f"bn.{t}"] = src[f"bn.{t}"]
    # encoder in/out
    for a, b in (("conv_in", "conv_in"), ("norm_out", "conv_norm_out"), ("conv_out", "conv_out")):
        dst[f"encoder.{a}.weight"] = src[f"encoder.{b}.weight"]
        dst[f"encoder.{a}.bias"] = src[f"encoder.{b}.bias"]
    # encoder down blocks (SAME index): 4 resolutions x 2 res-blocks
    for i in range(4):
        for j in range(2):
            _resnet(dst, f"encoder.down.{i}.block.{j}", src, f"encoder.down_blocks.{i}.resnets.{j}")
        if i != 3:
            dst[f"encoder.down.{i}.downsample.conv.weight"] = src[f"encoder.down_blocks.{i}.downsamplers.0.conv.weight"]
            dst[f"encoder.down.{i}.downsample.conv.bias"] = src[f"encoder.down_blocks.{i}.downsamplers.0.conv.bias"]
    _resnet(dst, "encoder.mid.block_1", src, "encoder.mid_block.resnets.0")
    _resnet(dst, "encoder.mid.block_2", src, "encoder.mid_block.resnets.1")
    _attn(dst, "encoder.mid.attn_1", src, "encoder.mid_block.attentions.0")
    # decoder in/out
    for a, b in (("conv_in", "conv_in"), ("norm_out", "conv_norm_out"), ("conv_out", "conv_out")):
        dst[f"decoder.{a}.weight"] = src[f"decoder.{b}.weight"]
        dst[f"decoder.{a}.bias"] = src[f"decoder.{b}.bias"]
    _resnet(dst, "decoder.mid.block_1", src, "decoder.mid_block.resnets.0")
    _resnet(dst, "decoder.mid.block_2", src, "decoder.mid_block.resnets.1")
    _attn(dst, "decoder.mid.attn_1", src, "decoder.mid_block.attentions.0")
    # decoder up blocks (REVERSED index): target up.i <-> source up_blocks.(3-i); 3 res-blocks
    for i in range(4):
        s = 3 - i
        for j in range(3):
            _resnet(dst, f"decoder.up.{i}.block.{j}", src, f"decoder.up_blocks.{s}.resnets.{j}")
        if i != 0:
            dst[f"decoder.up.{i}.upsample.conv.weight"] = src[f"decoder.up_blocks.{s}.upsamplers.0.conv.weight"]
            dst[f"decoder.up.{i}.upsample.conv.bias"] = src[f"decoder.up_blocks.{s}.upsamplers.0.conv.bias"]
    return dst


def _psnr(a, b):
    mse = torch.mean((a - b) ** 2).item()
    if mse <= 1e-12:
        return 99.0
    return 10.0 * torch.log10(torch.tensor(4.0 / mse)).item()  # data range 2 ([-1,1]) -> 2^2=4


def _textured_image(h=256, w=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij")
    img = torch.stack([
        torch.sin(6 * xx) * torch.cos(4 * yy),
        (xx + yy) / 2,
        torch.sin(12 * (xx ** 2 + yy ** 2)),
    ])
    # add sharp structure + fine texture so a scrambled mapping cannot fake a good PSNR
    img[:, h // 4:h // 2, w // 4:3 * w // 4] = 0.9
    fine = torch.nn.functional.interpolate(
        torch.rand(1, 3, h // 8, w // 8, generator=g), size=(h, w), mode="nearest")[0]
    img = (0.7 * img + 0.3 * (fine * 2 - 1)).clamp(-1, 1)
    return img.unsqueeze(0).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/models/flux2/FLUX.2-klein-base-4B/vae/diffusion_pytorch_model.safetensors")
    ap.add_argument("--out", default="/models/flux2/FLUX.2-klein-base-4B/ae.safetensors")
    ap.add_argument("--vae-dir", default="/models/flux2/FLUX.2-klein-base-4B/vae")
    args = ap.parse_args()

    from flux2.autoencoder import AutoEncoder, AutoEncoderParams

    src = load_file(args.src)
    dst = convert(src)
    print(f"converted {len(src)} diffusers params -> {len(dst)} flux2 params")

    ae = AutoEncoder(AutoEncoderParams()).eval()
    tgt_keys = set(ae.state_dict().keys())
    missing = tgt_keys - set(dst.keys())
    extra = set(dst.keys()) - tgt_keys
    if missing or extra:
        print(f"FAIL: key mismatch. missing={sorted(missing)[:6]} extra={sorted(extra)[:6]}", file=sys.stderr)
        return 1
    ae.load_state_dict(dst, strict=True)  # raises on any shape/name mismatch
    print("strict load into flux2 AutoEncoder: OK (251/251 keys)")

    x = _textured_image()
    with torch.no_grad():
        rec = ae.decode(ae.encode(x)).clamp(-1, 1)
    if not torch.isfinite(rec).all():
        print("FAIL: non-finite reconstruction.", file=sys.stderr)
        return 1
    psnr = _psnr(x, rec)
    print(f"flux2 AE encode->decode reconstruction PSNR: {psnr:.2f} dB")

    # Optional gold-standard cross-check vs the diffusers reference (if this diffusers has it).
    cross = None
    try:
        from diffusers import AutoencoderKLFlux2  # may not exist in older diffusers
        vae = AutoencoderKLFlux2.from_pretrained(args.vae_dir).eval()
        with torch.no_grad():
            enc = vae.encode(x)
            z = enc.latent_dist.mode() if hasattr(enc.latent_dist, "mode") else enc.latent_dist.mean
            rec2 = vae.decode(z).sample.clamp(-1, 1)
        cross = _psnr(rec, rec2)
        print(f"diffusers AutoencoderKLFlux2 reconstruction PSNR: {_psnr(x, rec2):.2f} dB")
        print(f"cross-check PSNR(flux2_recon, diffusers_recon): {cross:.2f} dB")
    except Exception as e:  # diffusers 0.34 may lack AutoencoderKLFlux2; PSNR check still governs
        print(f"(diffusers cross-check skipped: {type(e).__name__}: {str(e)[:120]})")

    ok = psnr >= 22.0 and (cross is None or cross >= 35.0)
    if not ok:
        print(f"FAIL: reconstruction below threshold (psnr={psnr:.2f}, cross={cross}).", file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    save_file(dst, args.out, metadata={"provenance": "converted from FLUX.2-klein-base-4B/vae (diffusers) to flux2-native AutoEncoder"})
    print(f"PASS: wrote flux2-native AE -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
