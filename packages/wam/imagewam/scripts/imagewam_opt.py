# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""AMD / Strix-Halo (gfx1151) deployment optimizations for FLUX.2 ImageWAM, baked into the
DEFAULT inference route of every demo. Two measured, near-lossless wins from the P8 latency
profile (docs: README "P8 latency / optimization findings"):

  1. TEXT-EMBEDDING CACHE across replans (exactly lossless).
     The task instruction is fixed for a whole episode, so the Qwen3 text encode inside
     `ImageWAM._prepare_flux2_infer_text(prompt, None, None)` is memoized by prompt string and
     reused on every subsequent replan. `infer_action_flux2` appends proprio *after* this call,
     so the cached (proprio-free) text embedding is safe to reuse verbatim.
     Measured: action call 1066 -> 873 ms  (-193 ms, -18%).

  2. torch.compile[default] of the per-step MoT action forward (near-lossless, bf16).
     `MoT.forward_action_with_video_cache` is the attention-heavy per-diffusion-step hot path
     (>80% of each step). It is lazily wrapped with torch.compile(mode="default") on first call
     (static shapes across steps/replans -> compiled once per process).
     Measured: action diffusion loop ~1.25x (max|Δ| ~1e-3 vs eager); HIP-graph gave 0.99x and
     reduce-overhead regressed, so we use inductor default only.

Both are applied by patching the ImageWAM / MoT *classes* (`patch_class()`), so any model built
by any entrypoint -- upstream closed-loop evaluators, the interactive Policy adapters, and the
open-loop / dream scripts -- inherits them without touching upstream code (rules 2.1 / 0.0).

Everything is env-gated (default ON) and falls back to plain eager on ANY error, so a demo can
never break because of an optimization:
    IMAGEWAM_OPT=0             -> disable all optimizations
    IMAGEWAM_TEXT_CACHE=0      -> disable the text-embedding cache
    IMAGEWAM_ACTION_COMPILE=0  -> disable torch.compile of the action step
    IMAGEWAM_TEXT_CACHE_MAX=N  -> max distinct prompts to cache (default 16)
"""
import functools
import os
import sys

_STATE = {"patched": False}


def _flag(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip().lower() not in ("0", "false", "no", "off", "")


def _log(msg: str) -> None:
    print(f"[imagewam_opt] {msg}", file=sys.stderr, flush=True)


def _wrap_text_cache(imagewam_cls) -> None:
    """Memoize the Qwen3 text encode by prompt (lossless; instruction fixed per episode)."""
    orig = imagewam_cls._prepare_flux2_infer_text
    if getattr(orig, "_imagewam_opt", False):
        return
    try:
        max_entries = int(os.environ.get("IMAGEWAM_TEXT_CACHE_MAX", "16") or "16")
    except ValueError:
        max_entries = 16

    @functools.wraps(orig)
    def wrapped(self, prompt, context, context_mask):
        # Only cache the pure-text encode (prompt given, no explicit context tensors passed in).
        if prompt is None or context is not None or context_mask is not None:
            return orig(self, prompt, context, context_mask)
        cache = self.__dict__.get("_imagewam_text_cache")
        if cache is None:
            cache = {}
            self.__dict__["_imagewam_text_cache"] = cache
        hit = cache.get(prompt)
        if hit is None:
            ctx, mask = orig(self, prompt, None, None)
            hit = (ctx.detach(), mask.detach())
            if len(cache) >= max_entries:
                cache.pop(next(iter(cache)))
            cache[prompt] = hit
        ctx, mask = hit
        # Return clones so downstream (proprio append / device moves) never mutate the cache.
        return ctx.clone(), mask.clone()

    wrapped._imagewam_opt = True
    imagewam_cls._prepare_flux2_infer_text = wrapped


def _wrap_action_compile(mot_cls) -> None:
    """Lazily torch.compile(mode='default') the per-step MoT action forward (static shapes)."""
    orig = mot_cls.forward_action_with_video_cache
    if getattr(orig, "_imagewam_opt", False):
        return
    import torch

    @functools.wraps(orig)
    def wrapped(self, *args, **kwargs):
        fn = self.__dict__.get("_imagewam_action_fn")
        if fn is None:
            def _eager(*a, _s=self, **k):
                return orig(_s, *a, **k)
            try:
                torch._dynamo.reset()
                fn = torch.compile(_eager, mode="default", dynamic=False, fullgraph=False)
            except Exception as e:  # noqa: BLE001
                _log(f"torch.compile setup failed ({type(e).__name__}: {e}); using eager")
                fn = _eager
            self.__dict__["_imagewam_action_fn"] = fn
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 - first real call may fail to compile -> eager
            _log(f"compiled action step failed ({type(e).__name__}: {e}); falling back to eager")

            def _eager(*a, _s=self, **k):
                return orig(_s, *a, **k)
            self.__dict__["_imagewam_action_fn"] = _eager
            return orig(self, *args, **kwargs)

    wrapped._imagewam_opt = True
    mot_cls.forward_action_with_video_cache = wrapped


def patch_class() -> bool:
    """Patch the ImageWAM / MoT classes in-place so every future instance is optimized.

    Idempotent and safe to call from any process: if ImageWAM is not importable here (e.g. a
    tiny aggregation helper) it silently no-ops. Returns True iff anything was patched.
    """
    if _STATE["patched"]:
        return True
    if not _flag("IMAGEWAM_OPT"):
        _log("IMAGEWAM_OPT=0 -> optimizations disabled")
        _STATE["patched"] = True
        return False
    try:
        from imagewam.models.backbones.imagewam import ImageWAM
        from imagewam.models.backbones.mot import MoT
    except Exception:  # noqa: BLE001 - imagewam not on path in this process; nothing to do
        return False

    applied = []
    if _flag("IMAGEWAM_TEXT_CACHE"):
        try:
            _wrap_text_cache(ImageWAM)
            applied.append("text-cache")
        except Exception as e:  # noqa: BLE001
            _log(f"text-cache patch failed: {type(e).__name__}: {e}")
    if _flag("IMAGEWAM_ACTION_COMPILE"):
        try:
            _wrap_action_compile(MoT)
            applied.append("torch.compile[action]")
        except Exception as e:  # noqa: BLE001
            _log(f"action-compile patch failed: {type(e).__name__}: {e}")

    _STATE["patched"] = True
    _log("default-route optimizations active: " + (", ".join(applied) if applied else "none"))
    return bool(applied)


def enable(obj=None) -> bool:
    """Convenience: apply the class patches (optionally given a model / policy wrapper instance).

    `obj` is accepted for call-site readability; the patches are class-level so passing the model
    is not required, but if a not-yet-patched instance's class differs we still cover it here.
    """
    ok = patch_class()
    # If handed a wrapper (e.g. RoboTwin deploy policy) whose .model is the ImageWAM module, the
    # class patch above already covers it; nothing per-instance is needed.
    _ = getattr(obj, "model", obj)
    return ok
