# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""ROCm (gfx1151 / Strix Halo) enablement shims for Cosmos3-Nano-Policy-DROID.

The upstream ``cosmos_framework`` attention frontend only ships CUDA-only backends
(cudnn / flash2 / flash3 / natten). On a ROCm build ``torch.version.cuda`` is None so
``get_arch_tag`` returns 0, ``get_backend_list`` returns [] and every attention call raises
``Could not find a compatible Attention backend``. This is a genuine hardware-port gap (rule 2.1),
so we register a PyTorch SDPA backend (``F.scaled_dot_product_attention`` -> AOTriton/CK on ROCm)
and make the selector prefer it.

Scope: the DROID policy config uses ``joint_attn_implementation="two_way"``, whose inference path
(single sample, ``inference_mode``) takes the dense branch of ``two_way_attention`` and calls the
standard ``attention()`` frontend with ``is_causal``/``causal_type`` only (no varlen, no LSE,
no natten). We still implement varlen + GQA + LSE so the backend is correct for the other paths.

Apply by importing this module once, after ``cosmos_framework`` is importable and before the first
forward:  ``import cosmos3_rocm_patches; cosmos3_rocm_patches.apply()``.
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F

_APPLIED = False


def _repeat_kv_heads(k: torch.Tensor, v: torch.Tensor, num_q_heads: int):
    """Expand [B,Hkv,S,D] -> [B,Hq,S,D] for grouped-query attention (backend-agnostic)."""
    num_kv_heads = k.shape[1]
    if num_kv_heads == num_q_heads:
        return k, v
    assert num_q_heads % num_kv_heads == 0, (
        f"num_q_heads={num_q_heads} not divisible by num_kv_heads={num_kv_heads}"
    )
    n_rep = num_q_heads // num_kv_heads
    k = k.repeat_interleave(n_rep, dim=1)
    v = v.repeat_interleave(n_rep, dim=1)
    return k, v


def _bottom_right_causal_mask(s_q: int, s_kv: int, device) -> torch.Tensor:
    """Additive bool mask for bottom-right aligned causal attention (SDPA convention: True=keep)."""
    q_idx = torch.arange(s_q, device=device).unsqueeze(1)
    k_idx = torch.arange(s_kv, device=device).unsqueeze(0)
    # query i (0-based from top) can see key j iff j <= i + (s_kv - s_q)
    return (k_idx <= (q_idx + (s_kv - s_q)))


def _sdpa_dense(q_bshd, k_bshd, v_bshd, *, is_causal, causal_type, scale, return_lse):
    from cosmos_framework.model.attention.masks import CausalType

    q = q_bshd.transpose(1, 2)  # [B,Hq,Sq,D]
    k = k_bshd.transpose(1, 2)  # [B,Hkv,Skv,D]
    v = v_bshd.transpose(1, 2)  # [B,Hkv,Skv,Dv]
    k, v = _repeat_kv_heads(k, v, q.shape[1])

    attn_mask = None
    causal_flag = False
    if is_causal:
        # TopLeft / DontCare == SDPA's native top-left aligned causal mask.
        if causal_type == getattr(CausalType, "BottomRight", object()):
            attn_mask = _bottom_right_causal_mask(q.shape[-2], k.shape[-2], q.device)
        else:
            causal_flag = True

    if return_lse:
        # Manual path (only exercised by non-two_way callers). Compute in fp32 for stable LSE.
        scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale  # [B,Hq,Sq,Skv]
        if causal_flag:
            s_q, s_kv = q.shape[-2], k.shape[-2]
            keep = torch.ones(s_q, s_kv, dtype=torch.bool, device=q.device).tril(diagonal=s_kv - s_q)
            scores = scores.masked_fill(~keep, float("-inf"))
        elif attn_mask is not None:
            scores = scores.masked_fill(~attn_mask, float("-inf"))
        lse = torch.logsumexp(scores, dim=-1)  # [B,Hq,Sq]
        probs = torch.softmax(scores, dim=-1).to(v.dtype)
        out = torch.matmul(probs, v)  # [B,Hq,Sq,Dv]
        out = out.transpose(1, 2).contiguous()  # [B,Sq,Hq,Dv]
        lse = lse.transpose(1, 2).contiguous()  # [B,Sq,Hq]
        return out, lse

    out = F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, is_causal=causal_flag, scale=scale
    )  # [B,Hq,Sq,Dv]
    return out.transpose(1, 2).contiguous()  # [B,Sq,Hq,Dv]


def _sdpa_varlen(query, key, value, *, is_causal, causal_type, scale,
                 cu_q, cu_kv, return_lse):
    # Sequence-packed layout: batch dim is 1, sequences concatenated along S.
    assert query.shape[0] == key.shape[0] == value.shape[0] == 1
    q = query[0]  # [Ntot_q,Hq,D]
    k = key[0]    # [Ntot_kv,Hkv,D]
    v = value[0]  # [Ntot_kv,Hkv,Dv]
    cu_q = [int(x) for x in cu_q.tolist()]
    cu_kv = [int(x) for x in cu_kv.tolist()]
    outs, lses = [], []
    for i in range(len(cu_q) - 1):
        qs = q[cu_q[i]:cu_q[i + 1]].unsqueeze(0)      # [1,sq,Hq,D]
        ks = k[cu_kv[i]:cu_kv[i + 1]].unsqueeze(0)    # [1,skv,Hkv,D]
        vs = v[cu_kv[i]:cu_kv[i + 1]].unsqueeze(0)    # [1,skv,Hkv,Dv]
        res = _sdpa_dense(qs, ks, vs, is_causal=is_causal, causal_type=causal_type,
                          scale=scale, return_lse=return_lse)
        if return_lse:
            o, l = res
            outs.append(o[0])
            lses.append(l[0])
        else:
            outs.append(res[0])
    out = torch.cat(outs, dim=0).unsqueeze(0)  # [1,Ntot_q,Hq,Dv]
    if return_lse:
        lse = torch.cat(lses, dim=0).unsqueeze(0)  # [1,Ntot_q,Hq]
        return out, lse
    return out


def sdpa_attention(query, key, value, is_causal=False, causal_type=None, scale=None,
                   cumulative_seqlen_Q=None, cumulative_seqlen_KV=None,
                   max_seqlen_Q=None, max_seqlen_KV=None, return_lse=False,
                   backend_kwargs=None, deterministic=False):
    """PyTorch SDPA attention backend matching the cosmos_framework backend contract.

    Inputs/outputs are heads-last ``[B,S,H,D]`` (BSHD in, BSHD out); optional LSE is ``[B,S,H]``.
    """
    scale = scale if scale is not None else query.shape[-1] ** -0.5
    if cumulative_seqlen_Q is not None:
        return _sdpa_varlen(query, key, value, is_causal=is_causal, causal_type=causal_type,
                            scale=scale, cu_q=cumulative_seqlen_Q, cu_kv=cumulative_seqlen_KV,
                            return_lse=return_lse)
    return _sdpa_dense(query, key, value, is_causal=is_causal, causal_type=causal_type,
                       scale=scale, return_lse=return_lse)


def sdpa_attention_check(query_shape=None, key_shape=None, value_shape=None, dtype=None,
                         device=None, requires_grad=False, is_causal=False, causal_type=None,
                         is_varlen=False, deterministic=False, raise_error=False):
    """SDPA is the universal fallback: accept everything the frontend already validated."""
    return True


def apply() -> None:
    """Register the SDPA backend and force the selector to use it. Idempotent."""
    global _APPLIED
    if _APPLIED:
        return

    # Run everything eager for bring-up: the attention selection error surfaced inside a
    # torch.compile graph, and complex Dynamo tracing over this MoT is out of scope for first light.
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    try:
        import torch._dynamo as _dynamo
        _dynamo.config.disable = True
    except Exception:
        pass

    from cosmos_framework.model.attention import backends as _backends
    from cosmos_framework.model.attention import frontend as _frontend

    _backends.BACKEND_CHECK_MAP["sdpa"] = sdpa_attention_check
    _frontend.BACKEND_MAP["sdpa"] = sdpa_attention

    def _get_backend_list(arch_tag):  # noqa: ARG001 - arch ignored; SDPA works on ROCm + CUDA
        return ["sdpa"]

    _backends.get_backend_list = _get_backend_list
    try:
        _backends.choose_backend.cache_clear()
    except Exception:
        pass

    _APPLIED = True
    print("[cosmos3_rocm_patches] SDPA attention backend registered; torch.compile disabled", flush=True)
