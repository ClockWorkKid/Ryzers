#!/usr/bin/env python3
"""Runtime optimization playbook patches for upstream Wan2.2 TI2V inference."""

from __future__ import annotations

import functools
import os
import time
from typing import Any, Callable

import torch

_APPLIED: set[str] = set()
_TIMINGS: dict[str, float] = {}


def get_timings() -> dict[str, float]:
    return dict(_TIMINGS)


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default) != "0"


def apply_torch_compile(pipe: Any) -> dict[str, Any]:
    """torch.compile knob (ROCm 7.14 inductor). Block-level compile keeps the
    python-heavy WanModel.forward eager while compiling the 30 transformer blocks
    (norm+linear+FFN); flash-attention custom ops become graph breaks. Lossless
    up to fp reassociation."""
    mode = os.environ.get("WAN22_COMPILE_MODE", "default")
    targets = os.environ.get("WAN22_COMPILE_TARGETS", "dit")
    flags: dict[str, Any] = {"torch_compile": True, "compile_mode": mode, "compile_targets": targets}
    torch._dynamo.config.cache_size_limit = max(64, getattr(torch._dynamo.config, "cache_size_limit", 8))
    torch._dynamo.config.suppress_errors = True  # fall back to eager on any unsupported region
    if "dit" in targets:
        for i, blk in enumerate(pipe.model.blocks):
            pipe.model.blocks[i] = torch.compile(blk, dynamic=True, mode=mode)
        flags["dit_blocks_compiled"] = len(pipe.model.blocks)
    if "vae" in targets:
        pipe.vae.model.decoder = torch.compile(pipe.vae.model.decoder, dynamic=True, mode=mode)
        flags["vae_decoder_compiled"] = True
    return flags


def _tome_merge_kv(k: torch.Tensor, v: torch.Tensor, ratio: float):
    """Training-free ToMe (bipartite soft matching) applied to K/V only.

    Queries stay full-resolution so 3D-RoPE positions and output length are
    preserved; we only shrink the key/value set each query attends to, cutting
    the O(Lq*Lk) attention scores. k,v: [b, L, n, d]. Returns merged k,v + k_lens.
    """
    b, L, n, d = k.shape
    r = int(L * ratio)
    with torch.no_grad():
        metric = k.reshape(b, L, n * d).float()
        metric = metric / (metric.norm(dim=-1, keepdim=True) + 1e-6)
        a, bset = metric[:, ::2], metric[:, 1::2]
        La = a.shape[1]
        r = min(r, La - 1)
        if r <= 0:
            return k, v, None
        scores = a @ bset.transpose(-1, -2)
        node_max, node_idx = scores.max(dim=-1)
        edge = node_max.argsort(dim=-1, descending=True)
        src = edge[:, :r]
        unm = edge[:, r:]
        dst = node_idx.gather(-1, src)

    def _g(t, idx):
        return t.gather(1, idx[:, :, None, None].expand(-1, -1, n, d))

    ka, kb = k[:, ::2], k[:, 1::2].clone()
    va, vb = v[:, ::2], v[:, 1::2].clone()
    dst_e = dst[:, :, None, None].expand(-1, -1, n, d)
    kb.scatter_add_(1, dst_e, _g(ka, src))
    vb.scatter_add_(1, dst_e, _g(va, src))
    cnt = torch.ones(b, kb.shape[1], 1, 1, device=k.device, dtype=k.dtype)
    cnt.scatter_add_(1, dst[:, :, None, None].expand(-1, -1, 1, 1),
                     torch.ones(b, r, 1, 1, device=k.device, dtype=k.dtype))
    kb, vb = kb / cnt, vb / cnt
    k_new = torch.cat([_g(ka, unm), kb], dim=1)
    v_new = torch.cat([_g(va, unm), vb], dim=1)
    klen = torch.full((b,), k_new.shape[1], dtype=torch.int32, device=k.device)
    return k_new, v_new, klen


def apply_dit_tome(model: Any, ratio: float) -> dict[str, Any]:
    """Patch WanSelfAttention to merge K/V by `ratio` (training-free token sparsity)."""
    from wan.modules.model import rope_apply
    from wan.modules.attention import flash_attention

    for blk in model.blocks:
        sa = blk.self_attn

        def sa_forward(x, seq_lens, grid_sizes, freqs, _sa=sa):
            b, s, n, dd = *x.shape[:2], _sa.num_heads, _sa.head_dim
            q = _sa.norm_q(_sa.q(x)).view(b, s, n, dd)
            k = _sa.norm_k(_sa.k(x)).view(b, s, n, dd)
            v = _sa.v(x).view(b, s, n, dd)
            q = rope_apply(q, grid_sizes, freqs)
            k = rope_apply(k, grid_sizes, freqs)
            k, v, klen = _tome_merge_kv(k, v, ratio)
            out = flash_attention(q, k, v, k_lens=klen, window_size=_sa.window_size)
            return _sa.o(out.flatten(2))

        sa.forward = sa_forward  # type: ignore[method-assign]
    return {"dit_tome_kv_ratio": ratio, "dit_tome_layers": len(model.blocks)}


def apply_playbook_opts(*, model: Any | None = None) -> dict[str, bool]:
    """Apply env-gated playbook patches; optionally bind caches to a loaded WanModel."""
    flags = {
        "text_kv_cache": _env_flag("WAN22_TEXT_KV_CACHE", "1"),
        "cfg_batch": _env_flag("WAN22_CFG_BATCH", "0"),
        "dit_timing": _env_flag("WAN22_DIT_TIMING", "1"),
    }
    if flags["text_kv_cache"] and model is not None:
        _patch_model_text_kv_cache(model)
    if flags["cfg_batch"]:
        _apply_cfg_batch()
    if flags["dit_timing"]:
        _apply_dit_vae_timing()
    return flags


def _patch_model_text_kv_cache(model: Any) -> None:
    """Cache text_embedding output + cross-attn K/V across denoise steps (bit-exact).

    Upstream ``WanModel.forward`` rebuilds the padded text tensor on *every* step,
    so we anchor the cache on the identity of the RAW context tensor (built once
    per generate() call and stable across all steps). ``model.forward`` stashes
    that key; the text-embedding wrapper then returns the *same* cached output
    object each step, which makes the downstream cross-attn context ptr stable too,
    so its K/V cache hits. Counters are exposed for verification.
    """
    if id(model) in _APPLIED:
        return

    text_emb_cache: dict[int, torch.Tensor] = {}
    cross_kv_cache: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
    stats = {"text_emb_hits": 0, "text_emb_miss": 0, "cross_kv_hits": 0, "cross_kv_miss": 0}
    state = {"ctx_key": 0}

    model._wan22_kv_stats = stats

    def _reset() -> None:
        text_emb_cache.clear()
        cross_kv_cache.clear()
        stats.update(text_emb_hits=0, text_emb_miss=0, cross_kv_hits=0, cross_kv_miss=0)

    model._wan22_reset_kv = _reset

    orig_model_forward = model.forward

    @functools.wraps(orig_model_forward)
    def keyed_forward(x, t, context, seq_len, y=None):
        raw = context[0] if isinstance(context, list) else context
        state["ctx_key"] = raw.data_ptr()
        return orig_model_forward(x, t, context, seq_len, y=y)

    model.forward = keyed_forward  # type: ignore[assignment]

    orig_text_forward = model.text_embedding.forward

    @functools.wraps(orig_text_forward)
    def cached_text_forward(x: torch.Tensor, *args, **kwargs):
        key = state["ctx_key"]
        hit = text_emb_cache.get(key)
        if hit is not None:
            stats["text_emb_hits"] += 1
            return hit
        stats["text_emb_miss"] += 1
        out = orig_text_forward(x, *args, **kwargs)
        text_emb_cache[key] = out
        return out

    model.text_embedding.forward = cached_text_forward  # type: ignore[method-assign]

    for block in model.blocks:
        cross = block.cross_attn
        orig_cross = cross.forward

        @functools.wraps(orig_cross)
        def cached_cross_forward(x, context, context_lens, _cross=cross, _orig=orig_cross):
            cache_key = (state["ctx_key"], id(_cross))
            b, n, d = x.size(0), _cross.num_heads, _cross.head_dim
            hit = cross_kv_cache.get(cache_key)
            if hit is not None:
                stats["cross_kv_hits"] += 1
                k, v = hit
            else:
                stats["cross_kv_miss"] += 1
                k = _cross.norm_k(_cross.k(context)).view(b, -1, n, d)
                v = _cross.v(context).view(b, -1, n, d)
                cross_kv_cache[cache_key] = (k, v)

            q = _cross.norm_q(_cross.q(x)).view(b, -1, n, d)
            from wan.modules.attention import flash_attention

            out = flash_attention(q, k, v, k_lens=context_lens)
            return _cross.o(out.flatten(2))

        cross.forward = cached_cross_forward  # type: ignore[method-assign]

    _APPLIED.add(id(model))


def _apply_cfg_batch() -> None:
    """Batch cond+uncond DiT forwards (CFG) into one pass with B=2."""
    if "cfg_batch" in _APPLIED:
        return

    import inspect
    import textwrap

    import wan.textimage2video as ti2v_mod

    original_t2v = ti2v_mod.WanTI2V.t2v
    # Dedent so the exec'd def sits at module scope, then splice the two sequential
    # CFG model() calls into a single batched B=2 forward (indentation-preserving).
    src = textwrap.dedent(inspect.getsource(original_t2v)).replace("def t2v(", "def t2v_cfg_batch(", 1)
    lines = src.split("\n")
    anchor = None
    for i, line in enumerate(lines):
        if line.strip().startswith("noise_pred_cond = self.model("):
            anchor = i
            break
    if anchor is None:
        return
    indent = lines[anchor][: len(lines[anchor]) - len(lines[anchor].lstrip())]
    new_block = [
        f"{indent}_cfg_t = timestep.repeat(2, 1)",
        f"{indent}_cfg_out = self.model([latents[0], latents[0]], t=_cfg_t, "
        f"context=[context[0], context_null[0]], seq_len=seq_len)",
        f"{indent}noise_pred_cond, noise_pred_uncond = _cfg_out[0], _cfg_out[1]",
    ]
    # The two upstream calls span 4 physical lines (each call wraps onto 2 lines).
    lines = lines[:anchor] + new_block + lines[anchor + 4:]
    patched_src = "\n".join(lines)
    namespace: dict[str, Any] = {}
    exec(patched_src, ti2v_mod.__dict__, namespace)  # noqa: S102
    cfg_fn = namespace["t2v_cfg_batch"]

    @functools.wraps(original_t2v)
    def t2v_entry(self, *args, **kwargs):
        guide_scale = kwargs.get("guide_scale", args[6] if len(args) > 6 else 5.0)
        if _env_flag("WAN22_CFG_BATCH", "0") and guide_scale != 1.0:
            return cfg_fn(self, *args, **kwargs)
        return original_t2v(self, *args, **kwargs)

    ti2v_mod.WanTI2V.t2v = t2v_entry
    _APPLIED.add("cfg_batch")


def _apply_dit_vae_timing() -> None:
    if "dit_timing" in _APPLIED:
        return

    import wan.textimage2video as ti2v_mod

    original_t2v = ti2v_mod.WanTI2V.t2v
    original_i2v = ti2v_mod.WanTI2V.i2v

    class _TimedModelProxy:
        def __init__(self, model: Any, on_call: Callable) -> None:
            self._model = model
            self._on_call = on_call

        def __call__(self, *args, **kwargs):
            return self._on_call(self._model, *args, **kwargs)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._model, name)

    def _wrap_generate(original: Callable) -> Callable:
        @functools.wraps(original)
        def timed(self, *args, **kwargs):
            _TIMINGS.clear()
            dit_s = 0.0
            vae_s = 0.0
            dit_t0: float | None = None
            real_model = self.model

            reset = getattr(real_model, "_wan22_reset_kv", None)
            if reset is not None:
                reset()

            def timed_call(model, *a, **kw):
                nonlocal dit_s, dit_t0
                if dit_t0 is None:
                    dit_t0 = time.perf_counter()
                out = model(*a, **kw)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                dit_s += time.perf_counter() - dit_t0
                dit_t0 = None
                return out

            vae = self.vae
            real_decode = vae.decode

            def timed_decode(zs):
                nonlocal vae_s
                t0 = time.perf_counter()
                out = real_decode(zs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                vae_s += time.perf_counter() - t0
                return out

            self.model = _TimedModelProxy(real_model, timed_call)
            vae.decode = timed_decode
            try:
                result = original(self, *args, **kwargs)
            finally:
                self.model = real_model
                vae.decode = real_decode
            _TIMINGS["dit_seconds"] = dit_s
            _TIMINGS["vae_decode_seconds"] = vae_s
            kv_stats = getattr(real_model, "_wan22_kv_stats", None)
            if kv_stats is not None:
                for k, v in kv_stats.items():
                    _TIMINGS[f"kv_{k}"] = v
            return result

        return timed

    ti2v_mod.WanTI2V.t2v = _wrap_generate(original_t2v)
    ti2v_mod.WanTI2V.i2v = _wrap_generate(original_i2v)
    _APPLIED.add("dit_timing")
