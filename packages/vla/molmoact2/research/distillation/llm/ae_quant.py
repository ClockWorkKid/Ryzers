"""Brevitas fake-quantization for the MolmoAct2 flow-matching ACTION EXPERT.

Mirrors the ViT student recipe (`../vit/quant/quant_student.py`,
AMD Pi0 SigLIP lineage) but re-targets it at the embedded
`ActionExpert` DiT stack instead of the standalone vision student.

Differences vs the ViT quantizer:
  * The AE attention (`ActionExpertSelfAttention` / `ActionExpertCrossAttention`)
    applies qk-RMSNorm + RoPE *between* the q/k/v projection and the SDPA, so we
    cannot collapse it into a single fused QSDPA block. Instead we keep each
    module's forward intact and only swap the inner `F.scaled_dot_product_attention`
    for a Brevitas-native `QuantScaledDotProductAttention` (per-instance method
    rebind). Detection is by class *name* so it works regardless of whether the AE
    was imported from the vendored HF copy or the lerobot overlay.
  * "defensive" mode keeps the small I/O layers (time/action embeddings, the two
    context K/V projections, and the final velocity head) at 8-bit while the 36
    transformer blocks go to the target W/A. `io_bits=None` quantizes everything
    to the target (the "worst-case" variant).

LSQ (PARAMETER_FROM_STATS learned per-channel step size) auto-activates for
weights at <=3-bit via the quantizer classes below. Fake-quant runs in float32.
"""
from __future__ import annotations

import types

import torch
from torch import nn

import brevitas.nn as qnn
from brevitas.graph.quantize import layerwise_quantize
from brevitas.inject.enum import ScalingImplType
from brevitas.quant.scaled_int import (
    Int8ActPerTensorFloat,
    Int8WeightPerChannelFloat,
)

# --- Bit-width-parametrised quantizers (subclass the 8-bit defaults) --------- #
# >=4-bit weights: STATS(MAX) per-channel scale. <=3-bit weights: learned-scale
# (LSQ) so QAT can move the step size off the outlier-dominated MAX init.


class Int4WeightPerChannelFloat(Int8WeightPerChannelFloat):
    bit_width = 4


class Int6WeightPerChannelFloat(Int8WeightPerChannelFloat):
    bit_width = 6


class Int3WeightPerChannelLSQ(Int8WeightPerChannelFloat):
    bit_width = 3
    scaling_impl_type = ScalingImplType.PARAMETER_FROM_STATS


class Int2WeightPerChannelLSQ(Int8WeightPerChannelFloat):
    bit_width = 2
    scaling_impl_type = ScalingImplType.PARAMETER_FROM_STATS


class Int4ActPerTensorFloat(Int8ActPerTensorFloat):
    bit_width = 4


class Int6ActPerTensorFloat(Int8ActPerTensorFloat):
    bit_width = 6


class Int3ActPerTensorFloat(Int8ActPerTensorFloat):
    bit_width = 3


class Int2ActPerTensorFloat(Int8ActPerTensorFloat):
    bit_width = 2


_WEIGHT_QUANT = {
    8: Int8WeightPerChannelFloat,
    6: Int6WeightPerChannelFloat,
    4: Int4WeightPerChannelFloat,
    3: Int3WeightPerChannelLSQ,
    2: Int2WeightPerChannelLSQ,
}
_ACT_QUANT = {
    8: Int8ActPerTensorFloat,
    6: Int6ActPerTensorFloat,
    4: Int4ActPerTensorFloat,
    3: Int3ActPerTensorFloat,
    2: Int2ActPerTensorFloat,
}

# I/O layer name substrings kept at higher precision in "defensive" mode.
_IO_SUBSTRINGS = ("time_embed", "action_embed", "context_k_proj",
                  "context_v_proj", "final_layer")
_ATTN_CLASSES = ("ActionExpertSelfAttention", "ActionExpertCrossAttention")


def _layer_map(weight_bits: int, act_bits: int) -> dict:
    kwargs = dict(
        input_quant=_ACT_QUANT[act_bits],
        weight_quant=_WEIGHT_QUANT[weight_bits],
        bias_quant=None,
        output_quant=None,
        return_quant_tensor=False,
    )
    return {nn.Linear: (qnn.QuantLinear, dict(kwargs))}


def _quant_attention(self, q, k, v, *, attn_mask=None, is_causal=False):
    """Replacement for ActionExpert*Attention._attention using QSDPA.

    Inputs q/k/v are [B, seq, heads, head_dim] (same convention the stock
    `_attention` expects); QSDPA wants [B, heads, seq, head_dim]. Scale matches
    the stock F.sdpa default (1/sqrt(head_dim)).
    """
    out = self.sdpa(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        attn_mask=attn_mask, is_causal=is_causal, scale=self.head_dim ** -0.5,
    )
    return out.transpose(1, 2).contiguous()


def _patch_ae_attn(ae: nn.Module, act_bits: int, classes=_ATTN_CLASSES) -> int:
    """Give the selected AE attention class(es) a QSDPA and rebind `_attention`."""
    signed = _ACT_QUANT[act_bits]
    n = 0
    for m in ae.modules():
        if type(m).__name__ in classes:
            m.sdpa = qnn.QuantScaledDotProductAttention(
                softmax_input_quant=signed,
                attn_output_weights_quant=signed,
                q_scaled_quant=signed,
                k_transposed_quant=signed,
                v_quant=signed,
                sdpa_output_quant=signed,
            )
            m._attention = types.MethodType(_quant_attention, m)
            n += 1
    return n


def _quantize_container(module: nn.Module, weight_bits: int, act_bits: int) -> nn.Module:
    return layerwise_quantize(module, compute_layer_map=_layer_map(weight_bits, act_bits))


# Ablation groups: which functional part of the AE to fake-quantize.
_BLOCK_GROUPS = ("self_attn", "cross_attn", "mlp", "modulation")
ALL_GROUPS = _BLOCK_GROUPS + ("io",)


def quantize_action_expert_(
    ae: nn.Module,
    weight_bits: int = 8,
    act_bits: int = 8,
    io_bits: int | None = 8,
    groups=None,
) -> nn.Module:
    """Fake-quantize the action expert in place.

    * The 36 transformer `blocks` -> (weight_bits, act_bits), split into the
      functional groups self_attn / cross_attn / mlp / modulation.
    * I/O layers (time/action embed, context k/v proj, final head): the "io"
      group; if ``io_bits`` is None they follow the target, otherwise pinned to
      (io_bits, io_bits) -- the "defensive" default keeps the velocity head sane.
    * self/cross attention SDPA -> QSDPA at ``act_bits`` (only for whichever of
      self_attn / cross_attn is selected).

    ``groups`` selects which parts to quantize (a subset of ``ALL_GROUPS``);
    ``None`` quantizes everything. Use a single group for sensitivity ablation.

    Run PTQ calibration (`brevitas.graph.calibrate.calibration_mode`) afterwards.
    AE must be float32 on entry.
    """
    sel = set(ALL_GROUPS) if groups is None else set(groups)
    bad = sel - set(ALL_GROUPS)
    if bad:
        raise ValueError(f"unknown quant groups {bad}; valid={ALL_GROUPS}")

    # 1) transformer blocks, per functional group
    blocks = getattr(ae, "blocks")
    for i in range(len(blocks)):
        blk = blocks[i]
        for gname in _BLOCK_GROUPS:
            if gname in sel and hasattr(blk, gname):
                setattr(blk, gname,
                        _quantize_container(getattr(blk, gname), weight_bits, act_bits))

    # 2) attention QSDPA (only for the selected attention group(s))
    attn_classes = tuple(
        c for c, g in (("ActionExpertSelfAttention", "self_attn"),
                       ("ActionExpertCrossAttention", "cross_attn")) if g in sel
    )
    n_attn = _patch_ae_attn(ae, act_bits=act_bits, classes=attn_classes) if attn_classes else 0

    # 3) I/O layers
    if "io" in sel:
        io_w = weight_bits if io_bits is None else io_bits
        io_a = act_bits if io_bits is None else io_bits
        ae.time_embed = _quantize_container(ae.time_embed, io_w, io_a)
        ae.final_layer = _quantize_container(ae.final_layer, io_w, io_a)
        tmp = nn.Sequential(ae.action_embed, ae.context_k_proj, ae.context_v_proj)
        tmp = _quantize_container(tmp, io_w, io_a)
        ae.action_embed, ae.context_k_proj, ae.context_v_proj = tmp[0], tmp[1], tmp[2]
        io_tag = "target" if io_bits is None else f"{io_bits}b"
    else:
        io_tag = "fp"

    print(f"[ae-quant] W{weight_bits}A{act_bits} groups={sorted(sel)} I/O@{io_tag}; "
          f"patched {n_attn} attention module(s)")
    return ae


def count_quant_linears(ae: nn.Module) -> dict:
    n_q = sum(1 for m in ae.modules() if isinstance(m, qnn.QuantLinear))
    n_fp = sum(1 for m in ae.modules()
               if isinstance(m, nn.Linear) and not isinstance(m, qnn.QuantLinear))
    n_sdpa = sum(1 for m in ae.modules()
                 if isinstance(m, qnn.QuantScaledDotProductAttention))
    return {"quant_linear": n_q, "fp_linear": n_fp, "qsdpa": n_sdpa}
