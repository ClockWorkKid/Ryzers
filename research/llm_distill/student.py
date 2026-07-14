"""Width-reduced LLM student for MolmoAct2 functional distillation.

``StudentTextModel`` is a drop-in replacement for ``model.model.transformer`` (the
36-layer ``MolmoAct2TextModel``). It mirrors that module's ``forward`` contract so BOTH
the training KV-collection path (``collect_layer_kv_states=True`` -> tuple of per-layer
(k,v)) and the inference path (``use_cache=True`` -> ``DynamicCache``) route through the
student and feed the frozen flow-matching action expert unchanged.

Key contract (must match the teacher exactly so the action expert's per-layer cross-attn
works):
- Consumes the teacher's fused ``inputs_embeds`` [B, N, 2560] (vision+text+state already
  fused upstream) via a 2560->hidden down-projection.
- Emits ``last_hidden_state`` [B, N, 2560] via a hidden->2560 up-projection (for the
  discrete action/LM head + ln).
- Emits per-layer KV [B, num_kv_heads, N, head_dim] captured **post-QK-norm, post-RoPE,
  PRE-GQA-repeat** -- byte-for-byte the same collection point as MolmoAct2Attention.
- Depth is FIXED at the teacher's num_layers (action expert is 1:1 per LLM layer).

Reduction comes from width (hidden / num_heads / intermediate); num_kv_heads*head_dim is
kept == teacher's kv_dim (1024) so the frozen action-expert context_{k,v}_proj fit natively.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- config
@dataclass
class StudentConfig:
    # ---- teacher-side contract (MolmoAct2-LIBERO); do NOT change for this teacher ----
    teacher_hidden: int = 2560           # fused inputs_embeds dim & last_hidden_state dim
    num_layers: int = 36                 # MUST equal teacher LLM layers (action-expert 1:1)
    num_kv_heads: int = 8                # kv_heads*head_dim must == teacher kv_dim (1024)
    head_dim: int = 128
    rope_theta: float = 5_000_000.0      # matches teacher (long fused-sequence regime)
    rms_eps: float = 1e-6
    use_qk_norm: bool = True             # Qwen3-style per-head RMSNorm on q,k (pre-RoPE)

    # ---- student width (the knob we sweep for the 2x/4x/8x reduction) ----
    hidden: int = 1024                   # Qwen3-0.6B width (anchor, ~6x reduction)
    num_heads: int = 16                  # query heads
    intermediate: int = 3072

    @property
    def kv_dim(self) -> int:
        return self.num_kv_heads * self.head_dim

    @property
    def q_dim(self) -> int:
        return self.num_heads * self.head_dim


# Named width presets for the reduction sweep. All keep 36 layers + 8 KV heads x 128
# (kv_dim 1024, native fit to the frozen action expert). ~reduction vs teacher decoder
# FLOPs is approximate (projection-only estimate).
PRESETS = {
    # anchor: Qwen3-0.6B width, ~6x
    "qwen06w": dict(hidden=1024, num_heads=16, intermediate=3072),
    # ~2x
    "w1792":   dict(hidden=1792, num_heads=28, intermediate=6784),
    # ~4x
    "w1280":   dict(hidden=1280, num_heads=20, intermediate=4864),
    # ~8x
    "w896":    dict(hidden=896,  num_heads=14, intermediate=2688),
}


def make_student_config(cfg: Optional[dict] = None) -> StudentConfig:
    cfg = dict(cfg or {})
    preset = cfg.pop("preset", None)
    base = dict(PRESETS.get(preset, {})) if preset else {}
    base.update(cfg)
    return StudentConfig(**base)


# --------------------------------------------------------------------------- layers
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dt)


def _rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    # q,k: [B, H, N, D]; cos/sin: [B, N, D]
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q = (q * cos) + (_rotate_half(q) * sin)
    k = (k * cos) + (_rotate_half(k) * sin)
    return q, k


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, theta: float):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, positions):
        # positions: [B, N] (long/float) -> cos,sin: [B, N, head_dim]
        freqs = positions[..., None].float() * self.inv_freq[None, None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


class Attention(nn.Module):
    """GQA attention mirroring MolmoAct2Attention's collection point.

    Collects (k, v) at [B, num_kv_heads, N, head_dim] AFTER qk-norm + RoPE and BEFORE the
    GQA head repeat -- the exact tensors the action expert consumes per layer.
    """

    def __init__(self, c: StudentConfig):
        super().__init__()
        self.c = c
        if c.num_heads % c.num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({c.num_heads}) must be divisible by num_kv_heads ({c.num_kv_heads})")
        if c.head_dim % 2 != 0:
            raise ValueError(f"head_dim ({c.head_dim}) must be even for RoPE")
        self.qkv = nn.Linear(c.hidden, c.q_dim + 2 * c.kv_dim, bias=False)
        self.o = nn.Linear(c.q_dim, c.hidden, bias=False)
        if c.use_qk_norm:
            self.q_norm = RMSNorm(c.head_dim, c.rms_eps)
            self.k_norm = RMSNorm(c.head_dim, c.rms_eps)

    def forward(self, x, cos, sin, attn_mask):
        B, N, _ = x.shape
        c = self.c
        qkv = self.qkv(x)
        q, k, v = torch.split(qkv, [c.q_dim, c.kv_dim, c.kv_dim], dim=-1)
        q = q.view(B, N, c.num_heads, c.head_dim)
        k = k.view(B, N, c.num_kv_heads, c.head_dim)
        v = v.view(B, N, c.num_kv_heads, c.head_dim)
        # Qwen3-style per-head qk-norm BEFORE transpose/RoPE (matches teacher order)
        if c.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q = q.transpose(1, 2)   # [B, Hq, N, D]
        k = k.transpose(1, 2)   # [B, Hkv, N, D]
        v = v.transpose(1, 2)   # [B, Hkv, N, D]
        q, k = apply_rope(q, k, cos, sin)
        # collection point: raw pre-repeat KV (what the action expert cross-attends to)
        k_collect, v_collect = k, v
        rep = c.num_heads // c.num_kv_heads
        kk = k.repeat_interleave(rep, dim=1)
        vv = v.repeat_interleave(rep, dim=1)
        out = F.scaled_dot_product_attention(q, kk, vv, attn_mask=attn_mask)
        out = out.transpose(1, 2).contiguous().view(B, N, c.q_dim)
        return self.o(out), (k_collect, v_collect)


class SwiGLU(nn.Module):
    def __init__(self, c: StudentConfig):
        super().__init__()
        self.gate_up = nn.Linear(c.hidden, 2 * c.intermediate, bias=False)
        self.down = nn.Linear(c.intermediate, c.hidden, bias=False)

    def forward(self, x):
        g, u = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(g) * u)


class DecoderLayer(nn.Module):
    def __init__(self, c: StudentConfig):
        super().__init__()
        self.attn_norm = RMSNorm(c.hidden, c.rms_eps)
        self.attn = Attention(c)
        self.ff_norm = RMSNorm(c.hidden, c.rms_eps)
        self.mlp = SwiGLU(c)

    def forward(self, x, cos, sin, attn_mask):
        h, kv = self.attn(self.attn_norm(x), cos, sin, attn_mask)
        x = x + h
        x = x + self.mlp(self.ff_norm(x))
        return x, kv


# --------------------------------------------------------------------------- model
class StudentTextModel(nn.Module):
    """Drop-in for ``model.model.transformer`` (MolmoAct2TextModel).

    forward mirrors MolmoAct2TextModel.forward's contract and returns an object with
    ``last_hidden_state`` [B,N,2560], ``past_key_values`` (tuple of per-layer (k,v) on the
    collect path, DynamicCache on the use_cache path), ``hidden_states``, ``attentions``.
    """

    def __init__(self, c: StudentConfig):
        super().__init__()
        self.c = c
        self.down_proj = nn.Linear(c.teacher_hidden, c.hidden, bias=False)
        self.layers = nn.ModuleList(DecoderLayer(c) for _ in range(c.num_layers))
        self.rope = RotaryEmbedding(c.head_dim, c.rope_theta)
        self.final_norm = RMSNorm(c.hidden, c.rms_eps)
        self.up_proj = nn.Linear(c.hidden, c.teacher_hidden, bias=False)

    # --- mask handling: accept whatever MolmoAct2Model.forward built and coerce to a
    #     4D additive float mask for SDPA (None -> plain causal via SDPA is_causal path). ---
    @staticmethod
    def _coerce_mask(attention_mask, dtype):
        if attention_mask is None:
            return None
        m = attention_mask
        if isinstance(m, dict):
            # mask mapping keyed by attention-type; the student is a single (full) type
            for key in ("full_attention", "full", "default"):
                if key in m:
                    m = m[key]
                    break
            else:
                m = next(iter(m.values()))
        if torch.is_tensor(m):
            if m.dtype == torch.bool:
                # True = keep -> additive 0 / -inf
                add = torch.zeros_like(m, dtype=dtype)
                add = add.masked_fill(~m, torch.finfo(dtype).min)
                return add
            return m.to(dtype)
        return None

    def _rope_cos_sin(self, position_ids, B, N, device, dtype):
        if position_ids is None:
            position_ids = torch.arange(N, device=device).unsqueeze(0).expand(B, N)
        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0).expand(B, N)
        cos, sin = self.rope(position_ids)
        return cos.to(dtype), sin.to(dtype)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        cache_position=None,
        **kwargs,
    ):
        if inputs_embeds is None:
            raise ValueError("StudentTextModel requires fused inputs_embeds (got input_ids only).")
        collect_layer_kv_states = bool(kwargs.pop("collect_layer_kv_states", False))
        B, N, _ = inputs_embeds.shape
        device, dtype = inputs_embeds.device, inputs_embeds.dtype

        if position_ids is None and cache_position is not None:
            position_ids = cache_position.unsqueeze(0) if cache_position.dim() == 1 else cache_position
        cos, sin = self._rope_cos_sin(position_ids, B, N, device, dtype)
        attn_mask = self._coerce_mask(attention_mask, dtype)

        x = self.down_proj(inputs_embeds)
        all_hidden = () if output_hidden_states else None
        collected = [] if collect_layer_kv_states else None

        # inference cache path: populate a DynamicCache exactly like MolmoAct2TextModel
        cache_obj = None
        if use_cache and not collect_layer_kv_states:
            from transformers.cache_utils import DynamicCache
            cache_obj = past_key_values if past_key_values is not None else DynamicCache()

        for li, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden += (x,)
            x, (k, v) = layer(x, cos, sin, attn_mask)
            if collect_layer_kv_states:
                collected.append((k, v))
            elif cache_obj is not None:
                cache_obj.update(k, v, li)

        x = self.final_norm(x)
        if output_hidden_states:
            all_hidden += (x,)
        last_hidden_state = self.up_proj(x)

        if collect_layer_kv_states:
            pkv = tuple(collected)
        elif cache_obj is not None:
            pkv = cache_obj
        else:
            pkv = past_key_values

        # Return a transformers-style output; fall back to a light namespace if the class
        # is unavailable (keeps Module-1 shape validation runnable outside the SIF).
        try:
            from transformers.modeling_outputs import BaseModelOutputWithPast
            return BaseModelOutputWithPast(
                last_hidden_state=last_hidden_state,
                past_key_values=pkv,
                hidden_states=all_hidden,
                attentions=None,
            )
        except Exception:
            from types import SimpleNamespace
            return SimpleNamespace(
                last_hidden_state=last_hidden_state,
                past_key_values=pkv,
                hidden_states=all_hidden,
                attentions=None,
            )


def build_student(cfg: Optional[dict] = None):
    c = make_student_config(cfg)
    return StudentTextModel(c), c


def student_flop_ratio(c: StudentConfig, teacher: Optional[StudentConfig] = None) -> float:
    """Approx per-token decoder FLOP reduction vs the teacher (projection-only estimate)."""
    def per_layer(hidden, nheads, nkv, hd, interm):
        qkv = hidden * (nheads * hd + 2 * nkv * hd)
        o = (nheads * hd) * hidden
        mlp = hidden * (2 * interm) + interm * hidden
        return qkv + o + mlp
    t = teacher or StudentConfig(hidden=2560, num_heads=32, intermediate=9728)
    tflops = per_layer(t.teacher_hidden if False else 2560, 32, 8, 128, 9728) * 36
    sflops = per_layer(c.hidden, c.num_heads, c.num_kv_heads, c.head_dim, c.intermediate) * c.num_layers
    return tflops / max(sflops, 1)


if __name__ == "__main__":
    student, c = build_student({"preset": "qwen06w"})
    n = sum(p.numel() for p in student.parameters())
    x = torch.randn(2, 48, c.teacher_hidden)
    out = student(inputs_embeds=x, collect_layer_kv_states=True)
    print(f"student params: {n/1e6:.1f}M  approx reduction: {student_flop_ratio(c):.1f}x")
    print(f"last_hidden_state: {tuple(out.last_hidden_state.shape)}")
    print(f"num kv layers: {len(out.past_key_values)}  k0: {tuple(out.past_key_values[0][0].shape)}")
