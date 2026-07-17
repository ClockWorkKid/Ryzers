#!/usr/bin/env python
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
Kernel-level parity probe for Latte's attention branches on gfx1151. Confirms that the `flash`
(torch SDPA) branch — with the ROCm port's `.transpose(1, 2)` fix (patches/rocm_port.py) — is
numerically identical to the upstream `math` branch, and demonstrates why the un-transposed
upstream `flash` reshape is wrong. Run inside the latte image: `python patches/opt/attn_parity_probe.py`.
"""
import sys, torch
sys.path.insert(0, "/repos/latte")
from models.latte import Attention

torch.manual_seed(0)
dev = "cuda"
B, N, C, H = 2, 256, 1152, 16
attn = Attention(C, num_heads=H, qkv_bias=True).to(dev).eval()
x = torch.randn(B, N, C, device=dev)


def run(mode):
    with torch.no_grad():
        Bx, Nx, Cx = x.shape
        qkv = attn.qkv(x).reshape(Bx, Nx, 3, attn.num_heads, Cx // attn.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv.unbind(0)
        if mode == "math":
            a = (q @ k.transpose(-2, -1)) * attn.scale
            a = a.softmax(dim=-1)
            o = (a @ v).transpose(1, 2).reshape(Bx, Nx, Cx)
        elif mode == "flash_upstream":  # upstream (WRONG on gfx1151): missing .transpose(1, 2)
            o = torch.nn.functional.scaled_dot_product_attention(q, k, v).reshape(Bx, Nx, Cx)
        elif mode == "flash_fixed":     # ROCm port fix: add .transpose(1, 2)
            o = torch.nn.functional.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(Bx, Nx, Cx)
        return attn.proj(o)


m = run("math")
for mode in ("flash_upstream", "flash_fixed"):
    o = run(mode)
    d = (o - m).abs()
    cos = torch.nn.functional.cosine_similarity(o.flatten(), m.flatten(), dim=0).item()
    print(f"{mode:16s}: max|Δ|={d.max().item():.4e} mean|Δ|={d.mean().item():.4e} cos={cos:.6f}")
