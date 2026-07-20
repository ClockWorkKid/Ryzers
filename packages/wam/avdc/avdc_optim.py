# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Opt-in inference optimizations for the AVDC video-diffusion policy on Strix Halo (gfx1151).

The sampling bottleneck is the 128x128x7-frame 3D-UNet: in the stock path MIOpen picks slow
immediate-mode 3D-conv kernels and the graph is unfused, giving ~5.4 s per DDIM step. Two env-gated
levers recover most of it (measured, 8-step gen):
  * AVDC_AMP=fp16|bf16  -> autocast the sampling loop (Trainer.sample has none). With good conv
                           kernels this alone gives ~12x; fp16 preferred on gfx1151.
  * AVDC_COMPILE=1      -> torch.compile the UNet (triton-fused kernels) -> ~31x (0.174 s/step).
Pair with MIOPEN_FIND_MODE=NORMAL so the flow model + any non-compiled convs also get tuned
kernels (cached in the mounted MIOpen dir). All optimizations are opt-in and numerically checked
by the quality gate (open-loop output + closed-loop success rate must be preserved).
"""
import os
import torch


def _truthy(v: str) -> bool:
    return str(v).lower() in ("1", "true", "yes", "on")


def _apply_sampling_amp():
    """Wrap GoalGaussianDiffusion.sample in autocast (idempotent, class-level)."""
    amp = os.environ.get("AVDC_AMP", "").lower()
    if amp not in ("fp16", "bf16"):
        return None
    dtype = torch.float16 if amp == "fp16" else torch.bfloat16
    from flowdiffusion.goal_diffusion import GoalGaussianDiffusion
    if not getattr(GoalGaussianDiffusion, "_avdc_amp_patched", False):
        _orig = GoalGaussianDiffusion.sample

        def sample(self, *a, **k):
            with torch.autocast("cuda", dtype=dtype):
                return _orig(self, *a, **k)

        GoalGaussianDiffusion.sample = sample
        GoalGaussianDiffusion._avdc_amp_patched = True
    return amp


def _compile_trainer(trainer) -> bool:
    """torch.compile the sampling (EMA) UNet and the base UNet. Shapes are fixed -> stable cache."""
    if not _truthy(os.environ.get("AVDC_COMPILE", "0")):
        return False
    mode = os.environ.get("AVDC_COMPILE_MODE", "default")
    try:
        trainer.ema.ema_model.model = torch.compile(trainer.ema.ema_model.model, mode=mode)
        trainer.model.model = torch.compile(trainer.model.model, mode=mode)
    except Exception as e:  # never let an optimization break the run
        print(f"[avdc_optim] torch.compile skipped ({e})")
        return False
    return True


def apply(trainer=None):
    """Apply env-gated optimizations. Call after building the video model, before sampling.

    Returns the list of applied optimization tags (for logging/metrics).
    """
    parts = []
    amp = _apply_sampling_amp()
    if amp:
        parts.append(f"amp={amp}")
    if trainer is not None and _compile_trainer(trainer):
        parts.append("compile=" + os.environ.get("AVDC_COMPILE_MODE", "default"))
    print("[avdc_optim]", ", ".join(parts) if parts else "baseline (fp32, no compile)")
    return parts
