# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Bake the two validated gfx1151 kernel-level speedups into the MimicGen policy build.

Measured on Radeon 8060S (gfx1151, ROCm 7.2.2), warm closed-loop step, sample_steps=40 (default):
    baseline (MIOpen conv + fp32 IDM) .......... 149.8 s   1.00x   (+ 350 s cold MIOpen autotune)
    + conv override (cudnn.enabled=False) ....... 17.0 s   8.80x
    + IDM bf16 (autocast) ....................... 15.5 s   9.66x   (IDM fidelity cos ~0.99998 vs fp32)

Both are pure kernel-level (no change to sample_steps / rollout horizon / any algorithm param):

  1. conv override  -- gfx1151 MIOpen has no tuned 3D conv solver (naive fallback dominates the WAN
     VAE conv3d AND the VGGT DPT Conv2d/ConvTranspose2d, and pays a ~350 s one-time autotune).
     `torch.backends.cudnn.enabled=False` routes every conv through ATen im2col/unfold + GEMM
     (rocBLAS/hipBLASLt) -- the "basic pytorch override" that won our cosmos3/strix conv microbenches.
  2. IDM bf16       -- the VGGT Jacobian IDM loads fp32 (build_policy only `.to(device)`). Its
     Jacobian is a *learned dense field* from ONE forward pass (not finite-difference), so bf16 is
     numerically safe. We wrap compute_jacobian in bf16 autocast (params stay fp32; gemms/attn run
     bf16 -> AOTriton flash) and cast the field back to fp32 for the downstream fp32 control math.

Env escape hatches (both default ON): VERA_DISABLE_CUDNN=0, VERA_IDM_BF16=0.
Idempotent (sentinel-guarded); never hard-fails the build. Applied against /repos/vera.
"""
import sys
from pathlib import Path

SENTINEL = "[ryzers-gfx1151-speedup]"

IMPORT_OLD = (
    "import torch\n"
    "from omegaconf import OmegaConf\n"
)
IMPORT_NEW = (
    "import torch\n"
    "from omegaconf import OmegaConf\n"
    "\n"
    "\n"
    "def _apply_gfx1151_speedups(policy):  # " + SENTINEL + "\n"
    "    \"\"\"Kernel-level gfx1151 speedups: conv override (cudnn off) + bf16 autocast IDM.\"\"\"\n"
    "    import os as _os\n"
    "    if _os.environ.get(\"VERA_DISABLE_CUDNN\", \"1\") == \"1\":\n"
    "        # MIOpen has no tuned 3D conv solver on gfx1151 -> naive fallback. Route conv through\n"
    "        # ATen im2col/unfold + GEMM (rocBLAS/hipBLASLt). ~8.8x on the closed-loop step.\n"
    "        torch.backends.cudnn.enabled = False\n"
    "        logging.info(\"[gfx1151-speedup] cudnn disabled (ATen conv im2col+GEMM)\")\n"
    "    if _os.environ.get(\"VERA_IDM_BF16\", \"1\") == \"1\":\n"
    "        dyn = getattr(policy, \"dynamics_model\", None)\n"
    "        model = getattr(dyn, \"model\", None)\n"
    "        if isinstance(model, torch.nn.Module) and hasattr(model, \"compute_jacobian\"):\n"
    "            _orig_cj = model.compute_jacobian\n"
    "            def _cj_bf16(input_obs, __orig=_orig_cj):\n"
    "                with torch.autocast(\"cuda\", dtype=torch.bfloat16):\n"
    "                    out = __orig(input_obs)\n"
    "                if isinstance(out, torch.Tensor):\n"
    "                    return out.float()\n"
    "                if isinstance(out, (tuple, list)):\n"
    "                    return type(out)(o.float() if isinstance(o, torch.Tensor) else o for o in out)\n"
    "                return out\n"
    "            model.compute_jacobian = _cj_bf16\n"
    "            logging.info(\"[gfx1151-speedup] IDM compute_jacobian wrapped in bf16 autocast\")\n"
    "    return policy\n"
)

RETURN_OLD = "    return MotionPolicyGripper(cfg, device=device)\n"
RETURN_NEW = (
    "    policy = MotionPolicyGripper(cfg, device=device)\n"
    "    return _apply_gfx1151_speedups(policy)  # " + SENTINEL + "\n"
)


def _patch(path: Path, old: str, new: str) -> str:
    if not path.exists():
        return f"SKIP (missing): {path}"
    text = path.read_text()
    if SENTINEL in text and new.strip() in text:
        return f"SKIP (already patched): {path}"
    if old not in text:
        return f"WARN (anchor not found, upstream drift?): {path}"
    path.write_text(text.replace(old, new, 1))
    return f"PATCHED: {path}"


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "/repos/vera")
    server = root / "vera" / "server" / "start_server_mimicgen.py"
    print(_patch(server, IMPORT_OLD, IMPORT_NEW))
    print(_patch(server, RETURN_OLD, RETURN_NEW))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
