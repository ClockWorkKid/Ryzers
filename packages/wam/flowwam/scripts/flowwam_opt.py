# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""AMD / Strix-Halo (gfx1151) deployment optimizations for FlowWAM's dual-stream video-DiT,
baked into the DEFAULT inference route of the closed-loop flow-action server. Two measured wins
from the P8 latency profile (docs: RUNTIME_OPTIMIZATION.md) that preserve output quality:

  1. STEP-INVARIANT CROSS-ATTN K/V + TEXT-EMBEDDING CACHE (bit-exact).
     The video-DiT denoise loop runs the SAME text `context` for all 25 diffusion steps (the
     server encodes the instruction once per episode and reuses it across replans). So:
       - `dit.text_embedding(context)` is memoized (returns the same tensor object per raw
         context), and
       - each block's cross-attention K,V = norm_k(k(ctx)), v(ctx) depend only on that context,
         yet are recomputed 2 streams x 25 steps = 50x/block. They are cached per (block, context)
         and only Q + attention + output projection are recomputed per call.
     Measured: bit-exact (max|Delta|=0 on rgb+flow latents), ~1.019x on the 25-step loop.

  2. torch.compile[default] of the hot dual-stream block fn (near-lossless, bf16).
     `_dual_stream_block_fn` is the per-block hot path; compiling it with Inductor fuses the
     fp32 norm / modulate / gate / FFN elementwise tail (~36% of the forward). Lazily wrapped on
     first call (static shapes across steps/replans -> compiled once per process).
     Measured: ~1.051x on the 25-step loop, near-lossless (cos 0.9999, max|Delta| ~1e-3 from bf16
     rounding). torch._dynamo.suppress_errors -> silent eager fallback on any graph issue.

Stacked (independent levers): ~1.07x on the dominant (~91%) video-DiT stage, quality-preserving.

Both are installed by wrapping the SHARED module function
`diffsynth.pipelines.wan_video_dual_stream.model_fn_wan_video_dual_stream`, so any process that
imports it -- the upstream flow-action server run verbatim through opt_launch.py -- inherits them
without editing upstream code (rules 2.1 / 0.0). The instance-level cache wrappers are installed
lazily on the FIRST forward (when the built `dit` is available), keeping this a pure runtime patch.

Everything is env-gated (default ON) and falls back to plain eager on ANY error, so a demo can
never break because of an optimization:
    FLOWWAM_OPT=0            -> disable all optimizations
    FLOWWAM_CACHE=0          -> disable the cross-attn K/V + text-embedding cache
    FLOWWAM_COMPILE=0        -> disable torch.compile of the dual-stream block fn
    FLOWWAM_COMPILE_MODE=... -> Inductor mode (default "default")

NOTE: the lossy flow-stream downsample lever (FLOW_DS, ~1.6x but drops closed-loop success at
DS=2) is deliberately NOT part of this default route; see RUNTIME_OPTIMIZATION.md.
"""
import os
import sys

_STATE = {"patched": False}


def _flag(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip().lower() not in ("0", "false", "no", "off", "")


def _log(msg: str) -> None:
    print(f"[flowwam_opt] {msg}", file=sys.stderr, flush=True)


def _make_text_emb_cache(torch):
    class _TextEmbCache(torch.nn.Module):
        """Memoize dit.text_embedding(context) by raw-context object identity (lossless)."""

        def __init__(self, mod):
            super().__init__()
            self.mod = mod
            self._key = None
            self._val = None

        def forward(self, x):
            if self._key is not x:  # new raw context (new episode/plan) -> recompute once
                self._val = self.mod(x)
                self._key = x
            return self._val

    return _TextEmbCache


def _make_cross_attn_cache(torch):
    class _CrossAttnCache(torch.nn.Module):
        """Cache text cross-attention K,V per context (they only depend on the constant context);
        recompute Q + attention + output per call. Bit-exact vs the upstream CrossAttention.forward
        for the text-only path. Kept eager (dynamo-disabled) so it composes with the compiled block."""

        def __init__(self, ca):
            super().__init__()
            self.ca = ca
            self._key = None
            self._k = None
            self._v = None

        @torch._dynamo.disable
        def forward(self, x, y):
            ca = self.ca
            if getattr(ca, "has_image_input", False):
                return ca(x, y)  # image cross-attn: fall back to upstream (uncached)
            if self._key is not y:  # context changed -> recompute K,V once
                self._k = ca.norm_k(ca.k(y))
                self._v = ca.v(y)
                self._key = y
            q = ca.norm_q(ca.q(x))
            return ca.o(ca.attn(q, self._k, self._v))

    return _CrossAttnCache


def _install_caches(torch, dit):
    if getattr(dit, "_flowwam_cache_installed", False):
        return
    TextEmbCache = _make_text_emb_cache(torch)
    CrossAttnCache = _make_cross_attn_cache(torch)
    dit.text_embedding = TextEmbCache(dit.text_embedding)
    n = 0
    for blk in dit.blocks:
        if not isinstance(getattr(blk, "cross_attn", None), CrossAttnCache):
            blk.cross_attn = CrossAttnCache(blk.cross_attn)
            n += 1
    dit._flowwam_cache_installed = True
    _log(f"cross-attn K/V + text-embedding cache installed on {n} blocks (bit-exact)")


def _install_compile(torch, dsp):
    if getattr(dsp, "_flowwam_compiled", False):
        return
    import torch._dynamo as dyn

    dyn.config.suppress_errors = True  # any graph break -> silent eager fallback
    mode = os.environ.get("FLOWWAM_COMPILE_MODE", "default") or "default"
    dsp._flowwam_block_fn_eager = dsp._dual_stream_block_fn
    dsp._dual_stream_block_fn = torch.compile(
        dsp._dual_stream_block_fn, mode=mode, dynamic=False
    )
    dsp._flowwam_compiled = True
    _log(f"torch.compile[{mode}] armed on _dual_stream_block_fn (compiles on first forward)")


def patch_class() -> bool:
    """Wrap the shared dual-stream model_fn so every forward installs the (idempotent) cache +
    compile optimizations. Safe to call from any process: no-ops if diffsynth is not importable
    here. Returns True iff the wrapper was installed."""
    if _STATE["patched"]:
        return True
    if not _flag("FLOWWAM_OPT"):
        _log("FLOWWAM_OPT=0 -> optimizations disabled")
        _STATE["patched"] = True
        return False
    try:
        import torch  # noqa: F401
        from diffsynth.pipelines import wan_video_dual_stream as dsp
    except Exception:  # noqa: BLE001 - diffsynth not on path in this process; nothing to do
        return False

    orig_model_fn = dsp.model_fn_wan_video_dual_stream
    if getattr(orig_model_fn, "_flowwam_opt", False):
        _STATE["patched"] = True
        return True

    do_cache = _flag("FLOWWAM_CACHE")
    do_compile = _flag("FLOWWAM_COMPILE")

    def wrapped(dit, flow_stream, *args, **kwargs):
        if do_cache:
            try:
                _install_caches(torch, dit)
            except Exception as e:  # noqa: BLE001
                _log(f"cache install failed ({type(e).__name__}: {e}); continuing eager")
        if do_compile:
            try:
                _install_compile(torch, dsp)
            except Exception as e:  # noqa: BLE001
                _log(f"compile arm failed ({type(e).__name__}: {e}); continuing eager")
        return orig_model_fn(dit, flow_stream, *args, **kwargs)

    wrapped._flowwam_opt = True
    dsp.model_fn_wan_video_dual_stream = wrapped

    applied = []
    if do_cache:
        applied.append("crossattn+text cache")
    if do_compile:
        applied.append("torch.compile[block]")
    _STATE["patched"] = True
    _log("default-route optimizations armed: " + (", ".join(applied) if applied else "none"))
    return True


def enable(obj=None) -> bool:
    """Convenience alias for patch_class(); `obj` accepted for call-site readability."""
    return patch_class()
