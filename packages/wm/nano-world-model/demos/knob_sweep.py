#!/usr/bin/env python
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
gfx1151 speedup knob sweep for the NanoWM DiT forward. For a given released checkpoint (--domain),
times ONE full DiT forward under a matrix of:
  * precision : fp32 / fp16 / bf16  (autocast; params stay fp32)
  * SDPA attn backend : default / math / flash / mem_efficient
The hipBLASLt-vs-rocBLAS GEMM path is process-global (env TORCH_BLAS_PREFER_HIPBLASLT), so run this
script twice (0 and 1) to A/B it. Emits knob_sweep.json in --out.
"""
import os, sys, json, time, argparse
import torch
from omegaconf import OmegaConf

sys.path.insert(0, "/repos/nanowm")
sys.path.insert(0, "/repos/nanowm/src")
from models import get_models
from latent_codecs import get_model_latent_channels, get_model_latent_size

try:
    from torch.nn.attention import sdpa_kernel, SDPBackend
    _SDPA = {
        "math": [SDPBackend.MATH],
        "flash": [SDPBackend.FLASH_ATTENTION],
        "mem_efficient": [SDPBackend.EFFICIENT_ATTENTION],
    }
except Exception:
    sdpa_kernel = None
    _SDPA = {}


def build(domain, results_dir):
    ckpt_dir = os.path.join(results_dir, domain)
    cfg = OmegaConf.load(os.path.join(ckpt_dir, "config.yaml"))
    OmegaConf.set_struct(cfg, False)
    OmegaConf.update(cfg, "latent_codec.kind", "sd_vae", force_add=True)
    OmegaConf.update(cfg, "latent_codec.model_path",
                     os.environ.get("VAE_MODEL_PATH", "stabilityai/sd-vae-ft-mse"), force_add=True)
    latent_size = get_model_latent_size(cfg); latent_ch = get_model_latent_channels(cfg)
    cfg.model.latent_size = latent_size
    model = get_models(cfg).to("cuda").eval()
    ckpt = os.path.join(ckpt_dir, "model.pt")
    if os.path.isfile(ckpt):
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = sd.get("model", sd)
        sd = {k[6:] if k.startswith("model.") else k: v for k, v in sd.items()}
        model.load_state_dict(sd, strict=False)
    B, T = 1, int(cfg.model.num_frames)
    act = int(cfg.dataset.spec.action_dim) * int(cfg.dataset.frame_interval)
    x = torch.randn(B, T, latent_ch, latent_size, latent_size, device="cuda")
    t = torch.randint(0, int(cfg.experiment.diffusion.diffusion_steps), (B, T), device="cuda")
    action = torch.randn(B, T, act, device="cuda") if cfg.model.use_action else None
    return model, dict(x=x, t=t, y=None, action=action, use_fp16=False), cfg


def timed(fn, iters=10, warmup=3):
    for _ in range(warmup):
        fn(); torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="dino_wm_pusht")
    ap.add_argument("--results_dir", default=os.environ.get("RESULTS_DIR", "/models/results"))
    ap.add_argument("--out", default="/outputs/knob_sweep")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_grad_enabled(False)

    model, inp, cfg = build(args.domain, args.results_dir)
    blas_pref = os.environ.get("TORCH_BLAS_PREFER_HIPBLASLT", "unset")

    precisions = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}
    backends = ["default"] + list(_SDPA.keys())
    grid = {}
    for pname, dt in precisions.items():
        grid[pname] = {}
        for bname in backends:
            def run(dt=dt, bname=bname):
                ctx_bk = sdpa_kernel(_SDPA[bname]) if (bname != "default" and sdpa_kernel) else None
                ctx_ac = torch.autocast("cuda", dtype=dt) if dt is not None else None
                if ctx_bk and ctx_ac:
                    with ctx_bk, ctx_ac: model(**inp)
                elif ctx_bk:
                    with ctx_bk: model(**inp)
                elif ctx_ac:
                    with ctx_ac: model(**inp)
                else:
                    model(**inp)
            try:
                grid[pname][bname] = round(timed(run), 2)
            except Exception as e:
                grid[pname][bname] = f"ERR:{type(e).__name__}"

    result = dict(
        domain=args.domain, arch=str(cfg.model.arch),
        num_frames=int(cfg.model.num_frames),
        blas_prefer_hipblaslt=blas_pref,
        device="gfx1151",
        dit_forward_ms=grid,
        note="ms per full DiT forward; rows=precision, cols=SDPA backend",
    )
    out = os.path.join(args.out, f"knob_sweep_{args.domain}_blas{blas_pref}.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    raise SystemExit(main())
