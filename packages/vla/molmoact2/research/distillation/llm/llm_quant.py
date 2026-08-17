"""Brevitas fake-quantization for the MolmoAct2 LLM BACKBONE (Qwen3-style).

Re-targets the action-expert quant recipe (`ae_quant.py`) at the frozen
`MolmoAct2TextModel` transformer (36 `MolmoAct2DecoderLayer`s, hidden 2560,
GQA 32q/8kv x128, SwiGLU intermediate 9728). ViT and the action expert stay
bf16; only the backbone is quantized.

Why this file exists separately from `ae_quant.py`:
  * The backbone packs q/k/v into a single fused ``att_proj`` (``nn.Linear``
    2560 -> 4096+1024+1024). The action expert cross-attends to the per-layer
    K/V produced by the k/v slice of that projection, so the *direct analog* of
    the action-expert's "keep cross-attention in full precision" fix is to keep
    the K/V slice full precision. To make that possible we SPLIT the fused
    ``att_proj`` into ``q_proj`` (2560->4096) and ``kv_proj`` (2560->2048) --
    numerically identical (a concatenation of two GEMMs) -- and quantize them
    independently.
  * ``MolmoAct2Attention.forward`` calls ``F.scaled_dot_product_attention``
    inline (no ``_attention`` seam like the action expert). To quantize the
    attention math we give the module its own Brevitas QSDPA and wrap its
    forward so that, *only for the duration of that instance's forward*, the
    module-global ``F.scaled_dot_product_attention`` routes through the QSDPA.
    This keeps the original forward (qk-norm ordering, RoPE, GQA repeat, cache
    update, per-layer KV collection) byte-for-byte intact.

Functional groups (a subset of ``ALL_GROUPS``):
  * ``attn_q``    : q_proj + attn_out linears (query/output projections).
  * ``attn_kv``   : the kv_proj slice (the K/V-producing path the action expert
                    reads).
  * ``attn_sdpa`` : the attention math itself (Brevitas QSDPA over the
                    query/key/value/softmax/attn-weights/output). Split out from
                    the projections so the sensitivity ablation can tell the
                    fragile attention *math* apart from the projection linears.
  * ``mlp``       : ff_proj + ff_out (SwiGLU).
  * ``io``        : token embedding / LM head. The released backbone's ``wte`` is
                    a Parameter-based ``MolmoAct2Embedding`` (not a quantizable
                    ``nn.Linear``/``nn.Embedding``) and the LM head is off the
                    continuous-action path, so this group is a no-op for this
                    checkpoint; kept for API parity with ``ae_quant``.

Selecting ``{attn_q, attn_kv, attn_sdpa, mlp}`` = the uniform scheme; dropping
whichever group the sensitivity ablation flags as fragile = the mixed-precision
scheme; a single group = a sensitivity-ablation cell.

LSQ (PARAMETER_FROM_STATS learned per-channel step size) auto-activates for
weights at <=3-bit via the quantizer classes reused from ``ae_quant``. Run in
bf16 (fp32-under-autocast NaNs during calibration, per the AE study).
"""
from __future__ import annotations

import torch
import torch.nn.functional as _F
from torch import nn

import brevitas.nn as qnn

# Reuse the exact weight/activation quantizer classes + layer map from the
# validated action-expert recipe (>=4-bit weights = STATS(MAX) per-channel;
# <=3-bit weights = learned-scale LSQ; signed per-tensor activations).
from ae_quant import _ACT_QUANT, _quantize_container  # noqa: E402


_ATTN_CLASS = "MolmoAct2Attention"
_DECODER_CLASSES = ("MolmoAct2DecoderLayer", "MolmoAct2PostNormDecoderLayer")

# fused att_proj output split: [q, k, v] = [4096, 1024, 1024] for the released
# MolmoAct2-LIBERO config (32*128, 8*128, 8*128). Read from the module at runtime
# so this stays correct if the config ever changes.
ALL_GROUPS = ("attn_q", "attn_kv", "attn_sdpa", "mlp", "io")
_ATTN_LINEAR_GROUPS = ("attn_q", "attn_kv")   # groups that need the att_proj split


class SplitQKVProj(nn.Module):
    """Drop-in for the fused ``att_proj``: concatenates q and kv projections.

    Numerically identical to the original fused ``nn.Linear`` (the downstream
    ``.split(fused_dims, dim=-1)`` sees the same tensor), but lets us quantize
    the query and key/value paths at different precisions.
    """

    def __init__(self, q_proj: nn.Module, kv_proj: nn.Module):
        super().__init__()
        self.q_proj = q_proj
        self.kv_proj = kv_proj

    def forward(self, x):
        return torch.cat([self.q_proj(x), self.kv_proj(x)], dim=-1)


def _split_att_proj_(attn: nn.Module) -> None:
    """Replace ``attn.att_proj`` (fused Linear) with a ``SplitQKVProj``.

    Idempotent. ``qkv_bias`` is False for this checkpoint so there is no bias to
    carry; we assert that to fail loudly if a future config enables it.
    """
    if isinstance(attn.att_proj, SplitQKVProj):
        return
    fused: nn.Linear = attn.att_proj
    q_dim = attn.num_heads * attn.head_dim
    kv_dim = attn.num_key_value_heads * attn.head_dim
    assert fused.out_features == q_dim + 2 * kv_dim, (
        f"unexpected att_proj out {fused.out_features} != {q_dim + 2 * kv_dim}")
    if fused.bias is not None:
        raise NotImplementedError("qkv_bias=True att_proj split not supported")
    in_dim = fused.in_features
    dev, dt = fused.weight.device, fused.weight.dtype
    q_proj = nn.Linear(in_dim, q_dim, bias=False, device=dev, dtype=dt)
    kv_proj = nn.Linear(in_dim, 2 * kv_dim, bias=False, device=dev, dtype=dt)
    with torch.no_grad():
        q_proj.weight.copy_(fused.weight[:q_dim])
        kv_proj.weight.copy_(fused.weight[q_dim:])
    attn.att_proj = SplitQKVProj(q_proj, kv_proj)


def _q_linear(lin: nn.Module, weight_bits: int, act_bits: int) -> nn.Module:
    """Fake-quantize a single ``nn.Linear`` (wrap -> layerwise_quantize -> unwrap)."""
    seq = nn.Sequential(lin)
    seq = _quantize_container(seq, weight_bits, act_bits)
    return seq[0]


def _wrap_attn_forward_with_qsdpa(attn: nn.Module, act_bits: int) -> None:
    """Give ``attn`` a Brevitas QSDPA and route its inline F.sdpa through it.

    The wrapper swaps ``torch.nn.functional.scaled_dot_product_attention`` only
    while this instance's forward runs (synchronous), so the original forward is
    untouched and no other module is affected. Idempotent.
    """
    if getattr(attn, "_qsdpa_wrapped", False):
        return
    signed = _ACT_QUANT[act_bits]
    attn.sdpa = qnn.QuantScaledDotProductAttention(
        softmax_input_quant=signed,
        attn_output_weights_quant=signed,
        q_scaled_quant=signed,
        k_transposed_quant=signed,
        v_quant=signed,
        sdpa_output_quant=signed,
    )
    orig_forward = attn.forward
    sdpa_mod = attn.sdpa

    def forward(*args, **kwargs):
        saved = _F.scaled_dot_product_attention

        def _q(query, key, value, attn_mask=None, dropout_p=0.0,
               is_causal=False, scale=None, **_kw):
            return sdpa_mod(query, key, value, attn_mask=attn_mask,
                            is_causal=is_causal, scale=scale)

        _F.scaled_dot_product_attention = _q
        try:
            return orig_forward(*args, **kwargs)
        finally:
            _F.scaled_dot_product_attention = saved

    attn.forward = forward
    attn._qsdpa_wrapped = True


def quantize_backbone_(
    transformer: nn.Module,
    weight_bits: int = 8,
    act_bits: int = 8,
    io_bits: int | None = 8,   # kept for API parity; io group is a no-op here
    groups=None,
) -> nn.Module:
    """Fake-quantize the LLM backbone transformer in place.

    ``groups`` selects which functional parts to quantize (subset of
    ``ALL_GROUPS``); ``None`` = everything (uniform). ``{attn_q, mlp}`` leaves
    the K/V path full precision (the protected scheme). Run PTQ calibration
    (``brevitas.graph.calibrate.calibration_mode``) afterwards.
    """
    sel = set(ALL_GROUPS) if groups is None else set(groups)
    bad = sel - set(ALL_GROUPS)
    if bad:
        raise ValueError(f"unknown quant groups {bad}; valid={ALL_GROUPS}")

    blocks = getattr(transformer, "blocks")
    n_q = n_kv = n_mlp = n_sdpa = 0
    for blk in blocks:
        if type(blk).__name__ not in _DECODER_CLASSES:
            continue
        attn = blk.self_attn

        # Split the fused att_proj whenever a projection side is quantized so the
        # q and k/v paths can be treated independently.
        if sel & set(_ATTN_LINEAR_GROUPS):
            _split_att_proj_(attn)

        if "attn_q" in sel:
            attn.att_proj.q_proj = _q_linear(attn.att_proj.q_proj, weight_bits, act_bits)
            attn.attn_out = _q_linear(attn.attn_out, weight_bits, act_bits)
            n_q += 1

        if "attn_kv" in sel:
            attn.att_proj.kv_proj = _q_linear(attn.att_proj.kv_proj, weight_bits, act_bits)
            n_kv += 1

        if "attn_sdpa" in sel:
            _wrap_attn_forward_with_qsdpa(attn, act_bits)
            n_sdpa += 1

        if "mlp" in sel:
            blk.mlp = _quantize_container(blk.mlp, weight_bits, act_bits)
            n_mlp += 1

    io_tag = "n/a"  # embedding is Parameter-based; not quantizable on this ckpt

    print(f"[llm-quant] W{weight_bits}A{act_bits} groups={sorted(sel)} io={io_tag}; "
          f"quantized blocks: attn_q={n_q} attn_kv={n_kv} mlp={n_mlp} qsdpa={n_sdpa}",
          flush=True)
    return transformer


def count_quant_linears(transformer: nn.Module) -> dict:
    n_q = sum(1 for m in transformer.modules() if isinstance(m, qnn.QuantLinear))
    n_fp = sum(1 for m in transformer.modules()
               if isinstance(m, nn.Linear) and not isinstance(m, qnn.QuantLinear))
    n_sdpa = sum(1 for m in transformer.modules()
                 if isinstance(m, qnn.QuantScaledDotProductAttention))
    n_split = sum(1 for m in transformer.modules() if isinstance(m, SplitQKVProj))
    return {"quant_linear": n_q, "fp_linear": n_fp, "qsdpa": n_sdpa,
            "split_qkv": n_split}
