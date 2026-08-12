# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Opt-in inference optimizations for GR-1 on ROCm/gfx1151 (phase 4).

Kept as a thin, env-toggled layer applied on top of the unmodified upstream model (rule 2.1: no
invasive edits to the vendored graph), so the same knobs drive both the benchmark harness
(demo_bench.py) and the closed-loop runner (demo_calvin.py), and so the optimizations remain an
opt-in wrapper for the eventual upstream PR. Toggles (host env -> container):

  GR1_AMP     off | bf16 | fp16     autocast dtype for the policy forward           (default off)
  GR1_SDPA    0 | 1                 use torch SDPA in the MAE ViT                    (default 0)
  GR1_FLASH   0 | 1                 pin the ROCm-native FLASH (AOTriton) SDPA backend(default 1)
  GR1_COMPILE 0 | 1                 torch.compile the policy                         (default 0)
  GR1_COMPILE_MODE  default|reduce-overhead|max-autotune                            (default default)

Notes:
 * MAE ViT-B (two batch-10 passes/step) dominates step cost, so SDPA targets its attention. The
   vendored ViT Attention returns (out, attn_weights) but the weights are discarded on the normal
   forward path (only get_last_selfattention, unused by GR-1, reads them), and SDPA's default scale
   equals head_dim**-0.5 == the module's scale, so this is numerically equivalent, just faster.
 * ROCm-native flash attention on gfx1151 (RDNA3.5) is delivered through torch SDPA's AOTriton
   FLASH backend -- the standalone flash_attn / composable-kernel library is CDNA-only and has no
   gfx1151 kernels. On-device probe: for fp16/bf16 the FLASH backend serves this ViT shape in
   ~0.18 ms vs ~3.1 ms for the MATH path, and torch's auto-select already picks FLASH; there is no
   flash kernel for fp32 ("No available kernel"). GR1_FLASH=1 therefore *pins* the flash backend
   (priority FLASH -> mem-efficient -> math) via torch.nn.attention.sdpa_kernel, so we deterministically
   run the ROCm flash kernel and never silently fall back to the slow math path; the mem-eff/math
   entries only catch the fp32 case where no flash kernel exists.
 * bf16 autocast additionally accelerates every matmul (resampler, GPT-2, linears) while keeping
   reductions/softmax/layernorm in fp32; fp16 (10 mantissa bits vs bf16's 7) better preserves
   closed-loop task success on gfx1151 (see RUNTIME_OPTIMIZATION.md).
"""
import contextlib
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

# Priority list: ROCm-native flash first, then the graceful fallbacks (mem-efficient handles the
# fp32 case where flash has no kernel; math is the last-resort correctness backend).
_FLASH_PRIORITY = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]

_AMP_MAP = {"off": None, "none": None, "no": None, "0": None,
            "fp16": torch.float16, "half": torch.float16,
            "bf16": torch.bfloat16, "bfloat16": torch.bfloat16}


def _amp_dtype():
    return _AMP_MAP.get((os.environ.get("GR1_AMP") or "off").lower(), None)


def _flag(name, default="0"):
    return (os.environ.get(name) or default).lower() in ("1", "true", "yes", "on")


def enable_vit_sdpa():
    """Monkeypatch the vendored MAE ViT Attention.forward to use torch SDPA (class-level, so it
    affects the already-built model_mae instance too). Idempotent."""
    import models.vision_transformer as vits
    if getattr(vits.Attention, "_sdpa_patched", False):
        return
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out, None
    vits.Attention.forward = forward
    vits.Attention._sdpa_patched = True


class _OptPolicy(nn.Module):
    """Wrap the policy so its forward runs under torch.autocast and/or the flash-pinned SDPA
    backend context, without touching upstream code."""
    def __init__(self, policy, amp_dtype, flash):
        super().__init__()
        self.policy = policy
        self.amp_dtype = amp_dtype
        self.flash = flash

    def forward(self, *args, **kwargs):
        amp_ctx = (torch.autocast(device_type="cuda", dtype=self.amp_dtype)
                   if self.amp_dtype is not None else contextlib.nullcontext())
        flash_ctx = sdpa_kernel(_FLASH_PRIORITY) if self.flash else contextlib.nullcontext()
        with flash_ctx, amp_ctx:
            return self.policy(*args, **kwargs)


def apply_optimizations(model, verbose=True):
    """Apply the env-selected optimizations to a GR1CalvinEvaluation instance in place.
    Returns the (label, config) for logging."""
    amp = _amp_dtype()
    sdpa = _flag("GR1_SDPA")
    flash = sdpa and _flag("GR1_FLASH", "1")  # pin ROCm flash backend only when SDPA is active
    do_compile = _flag("GR1_COMPILE")
    applied = []

    if sdpa:
        enable_vit_sdpa()
        applied.append("vit-sdpa(flash)" if flash else "vit-sdpa")
    if do_compile:
        mode = os.environ.get("GR1_COMPILE_MODE") or "default"
        model.policy = torch.compile(model.policy, mode=mode)
        applied.append(f"compile:{mode}")
    if amp is not None or flash:
        model.policy = _OptPolicy(model.policy, amp, flash)
        if amp is not None:
            applied.append("autocast:" + ("bf16" if amp is torch.bfloat16 else "fp16"))

    cfg = {"amp": ("bf16" if amp is torch.bfloat16 else "fp16" if amp is torch.float16 else "off"),
           "sdpa": bool(sdpa), "flash": bool(flash), "compile": bool(do_compile)}
    label = "+".join(applied) if applied else "fp32-baseline"
    if verbose:
        print(f"[gr1_optim] applied: {label}  ({cfg})")
    return label, cfg
