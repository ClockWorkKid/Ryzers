"""SDPA shim for flash_attn.

This package is installed via PYTHONPATH ahead of site-packages. The purpose is to
override the *top-level attention functions* (flash_attn_func, ...) with SDPA-backed
implementations that do not require a CUDA kernel. This way DreamZero's `import
flash_attn ; flash_attn.flash_attn_func(...)` calls run on ROCm via SDPA.

For *submodules* we don't override (e.g. `flash_attn.layers.rotary`,
`flash_attn.bert_padding` extras, ...), Python is directed to fall through to the
real flash_attn package installed in site-packages by extending this package's
`__path__`. The real package's submodules are mostly pure-torch helpers and are
safe to load. We deliberately avoid importing the real `flash_attn` *top-level*
because its own `__init__.py` tries to import `flash_attn_2_cuda` which is not
built on ROCm.

Coverage of the shim is intentionally minimal: extend it on demand, not eagerly.
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
from typing import Optional

import torch
import torch.nn.functional as F

__version__ = "0.0.0+sdpa-shim"

# Locate the real flash_attn package (sibling to us in site-packages) and extend
# this package's submodule search path to include it. We pin to the venv we're
# running in to avoid ambiguity.
_real_flash_attn_paths = []
for _site in sys.path:
    _candidate = os.path.join(_site, "flash_attn")
    if (
        os.path.isdir(_candidate)
        and os.path.realpath(_candidate) != os.path.realpath(os.path.dirname(__file__))
        and _candidate not in _real_flash_attn_paths
    ):
        _real_flash_attn_paths.append(_candidate)
__path__ = list(__path__) + _real_flash_attn_paths
# Note: __path__ ordering matters. Our overlay dir stays first, so any submodule
# we ship here (e.g. flash_attn_interface, bert_padding) wins over the real one.

_REQUESTED_BACKEND = os.environ.get("FLASH_ATTN_BACKEND", "sdpa").strip().lower()
_USE_NATIVE_ROCM = _REQUESTED_BACKEND in {"native", "rocm", "amd", "flash", "flash-attn"}
_native_flash_attn_interface = None


def _load_native_flash_attn_interface():
    """Load the installed ROCm flash-attn interface without importing this shim."""
    last_exc: Exception | None = None
    os.environ.setdefault("FLASH_ATTENTION_TRITON_AMD_ENABLE", "TRUE")
    for pkg_dir in _real_flash_attn_paths:
        interface_path = os.path.join(pkg_dir, "flash_attn_interface.py")
        if not os.path.exists(interface_path):
            continue
        spec = importlib.util.spec_from_file_location(
            "_dreamzero_native_flash_attn_interface",
            interface_path,
        )
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
            return module
        except Exception as exc:  # pragma: no cover - exercised on target ROCm host
            last_exc = exc
    if last_exc is not None:
        raise RuntimeError("FLASH_ATTN_BACKEND=native failed to load ROCm flash-attn") from last_exc
    raise RuntimeError(
        "FLASH_ATTN_BACKEND=native requested, but no installed flash_attn package "
        "was found behind the SDPA shim"
    )


if _USE_NATIVE_ROCM:
    _native_flash_attn_interface = _load_native_flash_attn_interface()
    __version__ = "2.8.4+rocm-native"


def _to_bnsd(x: torch.Tensor) -> torch.Tensor:
    if x.dim() != 4:
        raise ValueError(f"expected 4D (batch, seq, heads, dim), got shape={tuple(x.shape)}")
    return x.transpose(1, 2).contiguous()


def _from_bnsd(x: torch.Tensor) -> torch.Tensor:
    return x.transpose(1, 2).contiguous()


def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
):
    """flash_attn_func(q, k, v, ...) -> out

    Inputs are (batch, seqlen, nheads, headdim). Returns (batch, seqlen, nheads, headdim).
    Falls back to scaled_dot_product_attention. Sliding window and ALiBi are not implemented;
    raises if those features are requested.
    """
    if _native_flash_attn_interface is not None:
        return _native_flash_attn_interface.flash_attn_func(
            q, k, v,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            return_attn_probs=return_attn_probs,
        )

    if window_size != (-1, -1):
        raise NotImplementedError("flash_attn SDPA shim does not implement sliding window attention")
    if alibi_slopes is not None:
        raise NotImplementedError("flash_attn SDPA shim does not implement ALiBi slopes")
    if return_attn_probs:
        raise NotImplementedError("flash_attn SDPA shim cannot return attention probabilities")

    qb = _to_bnsd(q)
    kb = _to_bnsd(k)
    vb = _to_bnsd(v)

    out = F.scaled_dot_product_attention(
        qb, kb, vb,
        attn_mask=None,
        dropout_p=dropout_p if torch.is_grad_enabled() else 0.0,
        is_causal=causal,
        scale=softmax_scale,
    )
    return _from_bnsd(out)


def flash_attn_qkvpacked_func(
    qkv: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
):
    """qkv: (batch, seqlen, 3, nheads, headdim)."""
    if _native_flash_attn_interface is not None:
        return _native_flash_attn_interface.flash_attn_qkvpacked_func(
            qkv,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            return_attn_probs=return_attn_probs,
        )

    if qkv.dim() != 5 or qkv.size(2) != 3:
        raise ValueError(f"expected (batch, seqlen, 3, nheads, headdim), got shape={tuple(qkv.shape)}")
    q, k, v = qkv.unbind(dim=2)
    return flash_attn_func(
        q, k, v,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        return_attn_probs=return_attn_probs,
    )


def flash_attn_kvpacked_func(
    q: torch.Tensor,
    kv: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
):
    """kv: (batch, seqlen, 2, nheads, headdim)."""
    if _native_flash_attn_interface is not None:
        return _native_flash_attn_interface.flash_attn_kvpacked_func(
            q, kv,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            return_attn_probs=return_attn_probs,
        )

    if kv.dim() != 5 or kv.size(2) != 2:
        raise ValueError(f"expected (batch, seqlen, 2, nheads, headdim), got shape={tuple(kv.shape)}")
    k, v = kv.unbind(dim=2)
    return flash_attn_func(
        q, k, v,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        return_attn_probs=return_attn_probs,
    )


def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int = 0,
    max_seqlen_k: int = 0,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    **_unused_kwargs,
):
    """SDPA-backed varlen attention.

    Inputs (FA2 calling convention):
      q: (total_q, nheads, headdim)
      k: (total_k, nheads_k, headdim)
      v: (total_k, nheads_k, headdim_v)
      cu_seqlens_q: (B+1,) int32 -- 0, lq0, lq0+lq1, ..., total_q
      cu_seqlens_k: (B+1,) int32

    Returns:
      out: (total_q, nheads, headdim_v)

    Implementation: loop over batches, slice the packed sequences out,
    add a leading batch dim, run scaled_dot_product_attention, write back.
    For the single-batch / uniform-length case used by Wan2.1 cross-attn
    this collapses to one SDPA call.
    """
    if _native_flash_attn_interface is not None:
        return _native_flash_attn_interface.flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            return_attn_probs=return_attn_probs,
            **_unused_kwargs,
        )

    if window_size != (-1, -1):
        raise NotImplementedError("flash_attn SDPA shim does not implement sliding window for varlen")
    if alibi_slopes is not None:
        raise NotImplementedError("flash_attn SDPA shim does not implement ALiBi slopes for varlen")
    if return_attn_probs:
        raise NotImplementedError("flash_attn SDPA shim cannot return attention probabilities")
    if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
        raise ValueError(
            f"varlen expects 3D (total, nheads, headdim); got q={tuple(q.shape)} "
            f"k={tuple(k.shape)} v={tuple(v.shape)}"
        )

    cu_q = cu_seqlens_q.detach().to(dtype=torch.int64, device="cpu").tolist()
    cu_k = cu_seqlens_k.detach().to(dtype=torch.int64, device="cpu").tolist()
    if len(cu_q) != len(cu_k):
        raise ValueError(f"cu_seqlens_q ({len(cu_q)}) and cu_seqlens_k ({len(cu_k)}) length mismatch")
    B = len(cu_q) - 1

    out_shape = (q.shape[0], q.shape[1], v.shape[2])
    out = torch.empty(out_shape, dtype=q.dtype, device=q.device)

    for i in range(B):
        sq, eq = cu_q[i], cu_q[i + 1]
        sk, ek = cu_k[i], cu_k[i + 1]
        if eq == sq or ek == sk:
            continue
        # slice: (lq, n, d) -> (1, n, lq, d)
        qi = q[sq:eq].transpose(0, 1).unsqueeze(0).contiguous()
        ki = k[sk:ek].transpose(0, 1).unsqueeze(0).contiguous()
        vi = v[sk:ek].transpose(0, 1).unsqueeze(0).contiguous()
        oi = F.scaled_dot_product_attention(
            qi, ki, vi,
            attn_mask=None,
            dropout_p=dropout_p if torch.is_grad_enabled() else 0.0,
            is_causal=causal,
            scale=softmax_scale,
        )
        # (1, n, lq, dv) -> (lq, n, dv)
        out[sq:eq] = oi.squeeze(0).transpose(0, 1).contiguous()

    return out


flash_attn_unpadded_func = flash_attn_varlen_func
