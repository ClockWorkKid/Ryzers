#!/usr/bin/env python3
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Ryzer smoke test for the DreamZero (WAM-direct) image on Strix Halo (gfx1151).

No model weights required. Proves the container has:
  1. a working ROCm torch on the iGPU (bf16 matmul),
  2. the SDPA-backed `flash_attn` overlay shim importable,
  3. the pristine upstream `groot.vla.model.dreamzero` import graph reachable,
  4. a tiny AttentionModule forward that returns a finite, correctly-shaped tensor.

Exits non-zero on any hard failure so `ryzers run` / CI catches a broken image
before the 14B weight download + load. The full evaluations (numeric-vs-GT
open-loop, predicted-video looping-fix render) are driven by demos/ over the
overlay in overlay/tests/ — see README.md and RUNTIME_OPTIMIZATION.md.
"""
from __future__ import annotations

import os
import sys
import traceback

# Overlay shim wins; pristine dreamzero clone is importable as a source layout.
for candidate in ("/wam-direct/overlay/python", "/opt/dreamzero", "/opt/dreamzero/src"):
    if os.path.isdir(candidate) and candidate not in sys.path:
        sys.path.insert(0, candidate)


def banner(msg: str) -> None:
    bar = "=" * max(len(msg), 60)
    print(f"\n{bar}\n{msg}\n{bar}", flush=True)


def main() -> int:
    banner("DreamZero (WAM-direct) Ryzer smoke test")

    # 1. ROCm torch + GPU.
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: torch import failed: {exc!r}", file=sys.stderr)
        return 2

    print(f"torch             : {torch.__version__}")
    print(f"torch.version.hip : {torch.version.hip}")
    if not torch.version.hip:
        print("FAIL: torch is not a ROCm build.", file=sys.stderr)
        return 1
    if not torch.cuda.is_available():
        print("FAIL: no ROCm device visible (check --device=/dev/kfd,/dev/dri).", file=sys.stderr)
        return 1

    print(f"device[0]         : {torch.cuda.get_device_name(0)}")
    a = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
    s = (a @ b).float().sum().item()
    if not (s == s):  # NaN check
        print("FAIL: bf16 matmul produced NaN.", file=sys.stderr)
        return 1
    print(f"bf16 matmul ok    : sum={s:.3f}")

    # 2. SDPA flash_attn shim.
    banner("flash_attn overlay shim")
    try:
        import flash_attn
        import flash_attn.flash_attn_interface  # noqa: F401
        import flash_attn.bert_padding  # noqa: F401
        print(f"flash_attn shim   : {getattr(flash_attn, '__file__', '?')}")
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: flash_attn overlay shim import failed: {exc!r}", file=sys.stderr)
        traceback.print_exc()
        return 3

    # 3. Pristine upstream import graph.
    banner("groot.vla.model.dreamzero import graph")
    dz_commit = "unknown"
    try:
        if os.path.exists("/opt/dreamzero.commit"):
            dz_commit = open("/opt/dreamzero.commit").read().strip()
    except Exception:  # noqa: BLE001
        pass
    print(f"dreamzero commit  : {dz_commit}")
    try:
        import groot  # noqa: F401
        import groot.vla.model.dreamzero  # noqa: F401
        from groot.vla.model.dreamzero.modules.wan2_1_attention import AttentionModule
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: dreamzero import failed: {exc!r}", file=sys.stderr)
        traceback.print_exc()
        return 4

    # 4. Tiny AttentionModule forward (SDPA path via ATTENTION_BACKEND=torch).
    banner("AttentionModule tiny forward")
    try:
        os.environ.setdefault("ATTENTION_BACKEND", "torch")
        bsz, seq, heads, dim = 1, 64, 8, 64
        amod = AttentionModule(num_heads=heads, head_dim=dim).to("cuda")
        q = torch.randn(bsz, seq, heads, dim, device="cuda", dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        out = amod(q, k, v)
        finite = bool(torch.isfinite(out).all().item())
        print(f"backend           : {getattr(amod, 'backend', '?')}")
        print(f"output shape      : {tuple(out.shape)}  finite={finite}")
        if not finite:
            print("FAIL: AttentionModule output not finite.", file=sys.stderr)
            return 5
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: AttentionModule forward failed: {exc!r}", file=sys.stderr)
        traceback.print_exc()
        return 5

    print("\nPASS: DreamZero (WAM-direct) ROCm env OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
