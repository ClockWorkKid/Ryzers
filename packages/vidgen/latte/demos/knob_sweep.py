#!/usr/bin/env python
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
gfx1151 speedup knob sweep for the Latte DiT forward. For a class checkpoint (--dataset), times ONE
full DiT forward under a matrix of:
  * precision : fp32 / fp16 / bf16  (autocast; params stay fp32)
  * attention : math (manual matmul) / sdpa-default / sdpa-flash / sdpa-mem_efficient
The hipBLASLt-vs-rocBLAS GEMM path is process-global (TORCH_BLAS_PREFER_HIPBLASLT), so run twice
(0 and 1) to A/B it. Emits knob_sweep_<dataset>_blas<pref>.json in --out.
"""
import os, sys, json, time, argparse
import torch
from omegaconf import OmegaConf

sys.path.insert(0, "/repos/latte")
from models import get_models
from utils import find_model

try:
    from torch.nn.attention import sdpa_kernel, SDPBackend
    _SDPA = {
        "sdpa-default": None,
        "sdpa-flash": [SDPBackend.FLASH_ATTENTION],
        "sdpa-mem_efficient": [SDPBackend.EFFICIENT_ATTENTION],
    }
except Exception:
    sdpa_kernel = None
    _SDPA = {"sdpa-default": None}

CKPTS = {"ffs": "ffs.pt", "sky": "skytimelapse.pt", "taichi": "taichi-hd.pt", "ucf101": "ucf101.pt"}


def timed(fn, iters=10, warmup=3):
    for _ in range(warmup):
        fn(); torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


def set_attn_mode(model, mode):
    for m in model.modules():
        if hasattr(m, "attention_mode"):
            m.attention_mode = mode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=os.environ.get("DATASET", "sky"))
    ap.add_argument("--latte_dir", default=os.environ.get("LATTE_DIR", "/models/Latte"))
    ap.add_argument("--out", default="/outputs/knob_sweep")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_grad_enabled(False)

    cfg = OmegaConf.load(f"/repos/latte/configs/{args.dataset}/{args.dataset}_sample.yaml")
    latent = int(cfg.image_size) // 8
    model = get_models(OmegaConf.create({**cfg, "latent_size": latent})).to("cuda").eval()
    model.load_state_dict(find_model(os.path.join(args.latte_dir, CKPTS[args.dataset])))
    T = int(cfg.num_frames)
    extras = int(cfg.get("extras", 1) or 1)
    nc = cfg.get("num_classes", None)
    nc = int(nc) if nc else 0
    x = torch.randn(1, T, 4, latent, latent, device="cuda")
    t = torch.randint(0, 1000, (1,), device="cuda")
    y = torch.randint(0, nc, (1,), device="cuda") if (extras == 2 and nc) else None
    inp = dict(x=x, t=t, y=y)

    blas_pref = os.environ.get("TORCH_BLAS_PREFER_HIPBLASLT", "unset")
    precisions = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}
    attn_cols = ["math"] + list(_SDPA.keys())

    grid = {}
    for pname, dt in precisions.items():
        grid[pname] = {}
        for aname in attn_cols:
            set_attn_mode(model, "math" if aname == "math" else "flash")
            backend = _SDPA.get(aname) if aname != "math" else None

            def run(dt=dt, backend=backend, aname=aname):
                ctx_ac = torch.autocast("cuda", dtype=dt) if dt is not None else None
                ctx_bk = sdpa_kernel(backend) if (backend and sdpa_kernel) else None
                if ctx_ac and ctx_bk:
                    with ctx_ac, ctx_bk: model(**inp)
                elif ctx_ac:
                    with ctx_ac: model(**inp)
                elif ctx_bk:
                    with ctx_bk: model(**inp)
                else:
                    model(**inp)
            try:
                grid[pname][aname] = round(timed(run), 2)
            except Exception as e:
                grid[pname][aname] = f"ERR:{type(e).__name__}"

    result = dict(
        dataset=args.dataset, arch="Latte-XL/2", num_frames=T,
        blas_prefer_hipblaslt=blas_pref, device="gfx1151",
        dit_forward_ms=grid,
        note="ms per full DiT forward; rows=precision, cols=attention backend",
    )
    out = os.path.join(args.out, f"knob_sweep_{args.dataset}_blas{blas_pref}.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    raise SystemExit(main())
