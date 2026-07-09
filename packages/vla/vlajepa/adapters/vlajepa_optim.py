# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Runtime inference optimizations for the VLA-JEPA policy (shared by all sim adapters).

``apply_optimizations(model)`` patches an already-loaded ``baseframework`` in place with four
algorithmically near-lossless speedups, validated on gfx1151 to cut ``predict_action`` latency
~2.3-3.3x per call with no closed-loop task-success regression (200-episode LIBERO A/B). See
``RUNTIME_OPTIMIZATION.md`` for the full study, latency tables and parity numbers.

Applied (each guarded independently -- a structural mismatch skips only that opt, never crashes):
  1. bf16 DiT head      -- run the flow-matching head under autocast(bfloat16); the head otherwise
                           forces fp32 matmuls that bypass the matrix cores (biggest single win).
  2. skip lm_head       -- predict_action only reads the last hidden state; the 150k-vocab logits
                           projection is pure waste -> stub it (exactly lossless).
  3. cross-attn K/V cache -- the DiT cross-attends to the SAME VLM context on every Euler step, so
                           to_k/to_v are recomputed identically N times; memoize per layer per call.
  4. conv3d -> matmul   -- the vision patch-embed Conv3d has kernel==stride==full patch, i.e. it IS a
                           linear projection; a GEMM avoids the slow MIOpen conv3d path.

Diffusion steps are unchanged. ON by default; set ``VLAJEPA_NO_OPT=1`` to ship the stock fp32 path.
"""
import os

import torch
import torch.nn.functional as F

_OFF = ("1", "true", "yes", "on")


def _bf16_head(model, applied):
    head = model.action_model
    orig_predict = head.predict_action
    kv_caches = getattr(head, "_vlajepa_kv_caches", None)

    def _wrapped(*a, **k):
        if kv_caches:
            for c in kv_caches:
                c.clear()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = orig_predict(*a, **k)
        return out.float() if torch.is_tensor(out) else out

    head.predict_action = _wrapped
    applied.append("bf16_head")


def _skip_lm_head(model, applied):
    vlm = model.qwen_vl_interface.model
    lm_head = getattr(vlm, "lm_head", None)
    if lm_head is None:
        return

    def _lm_stub(*a, **k):
        x = k.get("input") if "input" in k else (a[0] if a else None)
        return x[..., :1]

    lm_head.forward = _lm_stub
    applied.append("skip_lm_head")


def _kv_cache(model, applied):
    head = model.action_model
    dit = head.model
    inner_dim = dit.inner_dim
    caches = []

    def memo(mod):
        orig = mod.forward
        cache = {}
        caches.append(cache)

        def fwd(x, *a, **k):
            hit = cache.get(id(x))
            if hit is not None and hit[0] is x:
                return hit[1]
            out = orig(x, *a, **k)
            cache[id(x)] = (x, out)
            return out

        mod.forward = fwd

    for blk in dit.transformer_blocks:
        if blk.attn1.to_k.in_features != inner_dim:  # cross-attn (K/V come from the VLM context)
            memo(blk.attn1.to_k)
            memo(blk.attn1.to_v)
    if caches:
        head._vlajepa_kv_caches = caches  # cleared each predict by the bf16 head wrapper
        applied.append("kv_cache")


def _conv_matmul(model, applied):
    vlm = model.qwen_vl_interface.model
    inner = getattr(vlm, "model", vlm)
    vision = getattr(inner, "visual", None) or getattr(vlm, "visual", None)
    if vision is None or not hasattr(vision, "patch_embed"):
        return
    pe = vision.patch_embed
    w_flat = pe.proj.weight.reshape(pe.embed_dim, -1).contiguous()
    bias = pe.proj.bias

    def _pe_fast(hidden_states):
        return F.linear(hidden_states.to(w_flat.dtype), w_flat, bias)

    pe.forward = _pe_fast
    applied.append("conv_matmul")


def apply_optimizations(model):
    """Patch ``model`` in place. Returns the list of opts applied (empty if disabled)."""
    if os.environ.get("VLAJEPA_NO_OPT", "").lower() in _OFF:
        print("[vlajepa_optim] disabled via VLAJEPA_NO_OPT -- stock fp32 inference path", flush=True)
        return []

    applied = []
    # kv_cache must register its caches on the head BEFORE the bf16 wrapper (which clears them).
    for name, fn in (("kv_cache", _kv_cache), ("bf16_head", _bf16_head),
                     ("skip_lm_head", _skip_lm_head), ("conv_matmul", _conv_matmul)):
        try:
            fn(model, applied)
        except Exception as e:  # noqa: BLE001 -- an opt must never break inference
            print(f"[vlajepa_optim] skipped {name}: {e}", flush=True)

    print(f"[vlajepa_optim] applied={applied} (set VLAJEPA_NO_OPT=1 to disable)", flush=True)
    return applied
