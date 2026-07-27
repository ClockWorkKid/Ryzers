"""Brevitas fake-quantization for the distilled MolmoAct2 vision-tower student.

Re-targets a standard SigLIP PTQ/QAT quantization recipe from an HF
``SiglipVisionModel`` to our custom ``distill.student.StudentEncoder`` (see
``distill/student.py``).

What changes vs a stock HF SigLIP quant recipe:
  * The student is NOT an HF model, so there is no ``SiglipAttention`` to patch.
    Instead we quantize ``nn.Linear`` (stem / pointwise / mlp / head) and
    ``nn.Conv2d`` (the depthwise conv in each ``ConvBlock``) via Brevitas
    ``layerwise_quantize``, then replace the ``AttnBlock``'s raw
    ``F.scaled_dot_product_attention`` with a Brevitas-native
    ``QuantScaledDotProductAttention`` (QSDPA) so the graph is dynamo-export-safe
    (QSDPA has a QONNX export_handler; the raw matmul path crashes dynamo).
  * The ``cnn`` variant has no attention -> pure QuantConv2d/QuantLinear, the
    cleanest FINN target (no QSDPA, no softmax).

Bit-widths are configurable per the plan's Pareto sweep {W8A8, W6A6, W4A8, W4A6};
the deployment target is **W4A6** (4-bit weights, 6-bit activations). Fake-quant
runs in float32 (exact integer numerics, no kernel speedup) -- the speedup comes
later from the FINN dataflow build.

FINN-feasibility notes (surfaced here, resolved at streamlining):
  * ``LayerNorm`` and ``GELU`` stay float under fake-quant; FINN streamlining must
    absorb/approximate them (LayerNorm -> mul+add after mean/var folding; GELU ->
    piecewise-linear / multithreshold). These are the main non-conv/non-matmul ops.
  * The head ``Linear(dim -> 2304)`` over 729 tokens is a large MVAU (the biggest
    single resource on-chip); watch it in the DSE.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

import brevitas.nn as qnn
from brevitas.graph.quantize import layerwise_quantize
from brevitas.quant.scaled_int import (
    Int8ActPerTensorFloat,
    Int8WeightPerChannelFloat,
    Uint8ActPerTensorFloat,
)

# Student building blocks (torch-only; safe to import wherever torch is present).
from distill.student import StudentEncoder, AttnBlock


# --- Bit-width-parametrised quantizers (subclass the 8-bit defaults) --------- #
class Int4WeightPerChannelFloat(Int8WeightPerChannelFloat):
    bit_width = 4


class Int6WeightPerChannelFloat(Int8WeightPerChannelFloat):
    bit_width = 6


class Int4ActPerTensorFloat(Int8ActPerTensorFloat):
    bit_width = 4


class Int6ActPerTensorFloat(Int8ActPerTensorFloat):
    bit_width = 6


class Uint4ActPerTensorFloat(Uint8ActPerTensorFloat):
    bit_width = 4


class Uint6ActPerTensorFloat(Uint8ActPerTensorFloat):
    bit_width = 6


_WEIGHT_QUANT = {8: Int8WeightPerChannelFloat, 6: Int6WeightPerChannelFloat, 4: Int4WeightPerChannelFloat}
_ACT_QUANT = {8: Int8ActPerTensorFloat, 6: Int6ActPerTensorFloat, 4: Int4ActPerTensorFloat}


class QuantAttnBlock(nn.Module):
    """Drop-in for ``distill.student.AttnBlock`` using Brevitas-native QSDPA.

    Reuses the (already weight+input quantised) ``qkv``/``proj``/``mlp`` linears
    and the float ``norm1``/``norm2`` from the wrapped block, and routes the
    attention through ``QuantScaledDotProductAttention`` so the exported QONNX is
    clean. Post-softmax weights are quantised SIGNED (FINN's QuantIdentityHandler
    only ingests signed Quant nodes on the identity activation fed by Softmax;
    signed spends the sign bit on unused negatives -> ~ (act_bits-1) effective
    bits, bump only this term if accuracy regresses).
    """

    def __init__(self, block: AttnBlock, act_bits: int = 6) -> None:
        super().__init__()
        self.heads = block.heads
        self.head_dim = block.head_dim
        self.norm1 = block.norm1
        self.qkv = block.qkv
        self.proj = block.proj
        self.norm2 = block.norm2
        self.mlp = block.mlp

        signed = _ACT_QUANT[act_bits]
        self.sdpa = qnn.QuantScaledDotProductAttention(
            softmax_input_quant=signed,
            attn_output_weights_quant=signed,   # FINN requires signed identity quant post-Softmax
            q_scaled_quant=signed,
            k_transposed_quant=signed,
            v_quant=signed,
            sdpa_output_quant=signed,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(b, n, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # F.scaled_dot_product_attention default scale = 1/sqrt(head_dim); pass it
        # explicitly so QSDPA (which applies scale on q internally) matches the fp path.
        o = self.sdpa(q, k, v, scale=self.head_dim ** -0.5)
        o = o.transpose(1, 2).reshape(b, n, c)
        x = x + self.proj(o)
        x = x + self.mlp(self.norm2(x))
        return x


def _patch_attn_blocks(model: nn.Module, act_bits: int) -> int:
    names = [n for n, m in model.named_modules() if isinstance(m, AttnBlock)]
    for name in names:
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p) if not p.isdigit() else parent[int(p)]
        leaf = parts[-1]
        block = getattr(parent, leaf) if not leaf.isdigit() else parent[int(leaf)]
        new = QuantAttnBlock(block, act_bits=act_bits)
        if leaf.isdigit():
            parent[int(leaf)] = new
        else:
            setattr(parent, leaf, new)
    return len(names)


def quantize_student_(student: StudentEncoder, weight_bits: int = 4, act_bits: int = 6) -> StudentEncoder:
    """Apply WxAy layerwise quantization (+ QSDPA attention) to a student encoder.

    Mutates and returns ``student``. Run PTQ calibration afterwards
    (``brevitas.graph.calibrate.calibration_mode``) before the quant scales are
    meaningful. Student must be float32 on entry.
    """
    weight_quant = _WEIGHT_QUANT[weight_bits]
    act_quant = _ACT_QUANT[act_bits]
    layer_kwargs = dict(
        input_quant=act_quant,
        weight_quant=weight_quant,
        bias_quant=None,
        output_quant=None,
        return_quant_tensor=False,
    )
    compute_layer_map = {
        nn.Linear: (qnn.QuantLinear, dict(layer_kwargs)),
        nn.Conv2d: (qnn.QuantConv2d, dict(layer_kwargs)),
    }
    student = layerwise_quantize(student, compute_layer_map=compute_layer_map)
    n_attn = _patch_attn_blocks(student, act_bits=act_bits)
    print(f"[quant] layerwise W{weight_bits}A{act_bits} applied; patched {n_attn} attn block(s)")
    return student


class SeamOut(nn.Module):
    """Wrap the fake-quant student so ``forward(patches)`` returns the bare seam
    tensor ``[B, N, seam_dim]`` -- a clean single-output graph for QONNX export."""

    def __init__(self, student: StudentEncoder) -> None:
        super().__init__()
        self.student = student

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        return self.student(patches)
