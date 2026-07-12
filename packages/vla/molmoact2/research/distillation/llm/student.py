"""Thin-twin LLM student for MolmoAct2 curriculum distillation (Run D).

Design (see README.md):
- Keeps 36 layers so the frozen flow-matching action expert's one-block-per-layer /
  per-layer-KV cross-attention conditioning still applies.
- Shrinks width (hidden 2560->208, interm 9728->832) for ~112x fewer decoder FLOPs.
- Mirrors the teacher block: pre-norm RMSNorm, GQA attention with RoPE + Qwen3-style
  QK-norm, SwiGLU MLP -- so per-layer features align 1:1 with the teacher for distillation.
- Consumes the teacher's fused ``inputs_embeds`` [B, N, 2560] (embedding lookup + additive
  vision fusion are cheap and shared) via a 2560->208 down-projection, and up-projects the
  final hidden 208->2560 (deployed; feeds discrete head + action-expert path).

Distillation heads (Stage-1 only, dropped at deploy) project student features to teacher
dims so per-layer hidden / KV regression losses can be computed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LLMStudentConfig:
    teacher_hidden: int = 2560          # MolmoAct2-LIBERO LLM hidden (input embeds dim)
    teacher_kv_dim: int = 1024          # 8 kv_heads * 128 head_dim (action-expert context in-dim)
    hidden: int = 208
    num_layers: int = 36                # MUST equal teacher LLM layers (action-expert constraint)
    num_heads: int = 8
    num_kv_heads: int = 8
    head_dim: int = 26                  # even -> RoPE pairs cleanly; q_dim = kv_dim = 208
    intermediate: int = 832
    rope_theta: float = 5_000_000.0     # matches teacher rope_theta
    rms_eps: float = 1e-6
    use_qk_norm: bool = True            # Qwen3-style per-head RMSNorm on q,k
    # which layers to place hidden-distill heads on (KV heads are on every layer since the
    # action expert consumes all 36); default = all layers.
    hidden_distill_layers: tuple = field(default_factory=lambda: tuple(range(36)))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * x.to(dt))


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
        # positions: [B, N]
        freqs = positions[..., None].float() * self.inv_freq[None, None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


class Attention(nn.Module):
    def __init__(self, c: LLMStudentConfig):
        super().__init__()
        self.c = c
        self.q_dim = c.num_heads * c.head_dim
        self.kv_dim = c.num_kv_heads * c.head_dim
        self.qkv = nn.Linear(c.hidden, self.q_dim + 2 * self.kv_dim, bias=False)
        self.o = nn.Linear(self.q_dim, c.hidden, bias=False)
        if c.use_qk_norm:
            self.q_norm = RMSNorm(c.head_dim, c.rms_eps)
            self.k_norm = RMSNorm(c.head_dim, c.rms_eps)

    def forward(self, x, cos, sin, attn_bias):
        B, N, _ = x.shape
        c = self.c
        qkv = self.qkv(x)
        q, k, v = torch.split(qkv, [self.q_dim, self.kv_dim, self.kv_dim], dim=-1)
        q = q.view(B, N, c.num_heads, c.head_dim).transpose(1, 2)
        k = k.view(B, N, c.num_kv_heads, c.head_dim).transpose(1, 2)
        v = v.view(B, N, c.num_kv_heads, c.head_dim).transpose(1, 2)
        if c.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k = apply_rope(q, k, cos, sin)
        # raw (pre-repeat) KV are the distillation targets that the action expert consumes
        k_raw, v_raw = k, v
        rep = c.num_heads // c.num_kv_heads
        kk = k.repeat_interleave(rep, dim=1)
        vv = v.repeat_interleave(rep, dim=1)
        out = F.scaled_dot_product_attention(q, kk, vv, attn_mask=attn_bias)
        out = out.transpose(1, 2).contiguous().view(B, N, self.q_dim)
        return self.o(out), (k_raw, v_raw)


class SwiGLU(nn.Module):
    def __init__(self, c: LLMStudentConfig):
        super().__init__()
        self.gate_up = nn.Linear(c.hidden, 2 * c.intermediate, bias=False)
        self.down = nn.Linear(c.intermediate, c.hidden, bias=False)

    def forward(self, x):
        g, u = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(g) * u)


class DecoderLayer(nn.Module):
    def __init__(self, c: LLMStudentConfig):
        super().__init__()
        self.attn_norm = RMSNorm(c.hidden, c.rms_eps)
        self.attn = Attention(c)
        self.ff_norm = RMSNorm(c.hidden, c.rms_eps)
        self.mlp = SwiGLU(c)

    def forward(self, x, cos, sin, attn_bias):
        h, kv = self.attn(self.attn_norm(x), cos, sin, attn_bias)
        x = x + h
        x = x + self.mlp(self.ff_norm(x))
        return x, kv


class LLMStudent(nn.Module):
    """Thin-twin student. forward returns per-layer hidden states, per-layer raw KV, and
    the 2560-d up-projected final hidden. Distillation heads (project to teacher dims) are
    exposed separately so they can be excluded from the deployed model."""

    def __init__(self, c: LLMStudentConfig):
        super().__init__()
        self.c = c
        self.down_proj = nn.Linear(c.teacher_hidden, c.hidden, bias=False)
        self.layers = nn.ModuleList(DecoderLayer(c) for _ in range(c.num_layers))
        self.rope = RotaryEmbedding(c.head_dim, c.rope_theta)
        self.final_norm = RMSNorm(c.hidden, c.rms_eps)
        self.up_proj = nn.Linear(c.hidden, c.teacher_hidden, bias=False)

    def forward(self, inputs_embeds, attention_bias=None, positions=None):
        B, N, _ = inputs_embeds.shape
        if positions is None:
            positions = torch.arange(N, device=inputs_embeds.device).unsqueeze(0).expand(B, N)
        cos, sin = self.rope(positions)
        cos = cos.to(inputs_embeds.dtype)
        sin = sin.to(inputs_embeds.dtype)
        x = self.down_proj(inputs_embeds)
        hidden_states = []
        kv_states = []
        for layer in self.layers:
            hidden_states.append(x)             # pre-layer input (teacher output_hidden_states convention)
            x, kv = layer(x, cos, sin, attention_bias)
            kv_states.append(kv)
        x = self.final_norm(x)
        hidden_states.append(x)                 # post-final-norm (matches teacher last hidden position)
        final_2560 = self.up_proj(x)
        return {
            "hidden_states": hidden_states,      # list length num_layers+1, dim=hidden
            "kv_states": kv_states,              # list length num_layers, each (k,v) [B,kvH,N,hd]
            "last_hidden_state": final_2560,     # [B, N, teacher_hidden]
            "last_hidden_student": x,            # [B, N, hidden]
        }


class DistillHeads(nn.Module):
    """Stage-1-only projectors mapping student features -> teacher dims for regression.
    Dropped at deploy (final up_proj + retrained action-expert KV projections handle deploy)."""

    def __init__(self, c: LLMStudentConfig):
        super().__init__()
        self.c = c
        # one hidden head per selected layer (project 208 -> 2560)
        self.hidden_heads = nn.ModuleDict(
            {str(i): nn.Linear(c.hidden, c.teacher_hidden, bias=False)
             for i in c.hidden_distill_layers}
        )
        # per-layer k/v heads project raw student KV (kvH*hd=208) -> teacher kv (1024)
        kv_in = c.num_kv_heads * c.head_dim
        self.k_heads = nn.ModuleList(nn.Linear(kv_in, c.teacher_kv_dim, bias=False)
                                     for _ in range(c.num_layers))
        self.v_heads = nn.ModuleList(nn.Linear(kv_in, c.teacher_kv_dim, bias=False)
                                     for _ in range(c.num_layers))

    def project_hidden(self, layer_idx, h):
        return self.hidden_heads[str(layer_idx)](h)

    def project_kv(self, layer_idx, k_raw, v_raw):
        B, H, N, D = k_raw.shape
        kf = k_raw.transpose(1, 2).reshape(B, N, H * D)
        vf = v_raw.transpose(1, 2).reshape(B, N, H * D)
        return self.k_heads[layer_idx](kf), self.v_heads[layer_idx](vf)


def build_student(cfg: Optional[dict] = None):
    c = LLMStudentConfig(**(cfg or {}))
    return LLMStudent(c), DistillHeads(c), c


if __name__ == "__main__":
    student, heads, c = build_student()
    n_s = sum(p.numel() for p in student.parameters())
    n_h = sum(p.numel() for p in heads.parameters())
    x = torch.randn(2, 64, c.teacher_hidden)
    out = student(x)
    print(f"student params: {n_s/1e6:.1f}M  (distill heads: {n_h/1e6:.1f}M, dropped at deploy)")
    print(f"hidden_states: {len(out['hidden_states'])} x {tuple(out['hidden_states'][0].shape)}")
    print(f"kv_states: {len(out['kv_states'])} x k{tuple(out['kv_states'][0][0].shape)}")
    print(f"last_hidden_state (teacher dim): {tuple(out['last_hidden_state'].shape)}")
    hp = heads.project_hidden(0, out['last_hidden_student'])
    kp, vp = heads.project_kv(0, *out['kv_states'][0])
    print(f"projected hidden: {tuple(hp.shape)}  projected k: {tuple(kp.shape)}")
