"""Self-contained distilled student vision encoder for in-container injection.

Mirrors research/vit_distill/distill/{student,config}.py exactly (hybrid variant,
dim=256, 4 conv blocks + 3 attn blocks, 8 heads) so the M3 checkpoint
`hybrid_full.pt` loads with strict=True. Kept dependency-free (torch only) so it
can be bind-mounted into the lerobot eval SIF without the vit_distill package.

Contract (matches MolmoAct2VisionBackbone.encode_image):
    INPUT  normalized patches [B, num_crops, 729, 588], bf16, values in [-1, 1]
    OUTPUT seam [B, num_crops, 729, 2304]  (concat of ViT layers 24 || 18)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class StudentConfig:
    variant: str = "hybrid"
    grid: int = 27
    in_pixels: int = 588
    dim: int = 256
    seam_dim: int = 2304
    num_conv_blocks: int = 4
    conv_kernel: int = 3
    num_attn_blocks: int = 3
    attn_heads: int = 8
    attn_mlp_ratio: float = 2.0
    head_hidden: int = 0

    @property
    def num_tokens(self) -> int:
        return self.grid * self.grid


class ConvBlock(nn.Module):
    def __init__(self, dim: int, kernel: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dw = nn.Conv2d(dim, dim, kernel, padding=kernel // 2, groups=dim)
        self.pw = nn.Linear(dim, dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, grid: int) -> torch.Tensor:
        b, n, c = x.shape
        h = self.norm(x)
        h = h.transpose(1, 2).reshape(b, c, grid, grid)
        h = self.dw(h)
        h = h.reshape(b, c, n).transpose(1, 2)
        h = self.act(self.pw(h))
        return x + h


class AttnBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(round(mlp_ratio * dim))
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(b, n, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        o = F.scaled_dot_product_attention(q, k, v)
        o = o.transpose(1, 2).reshape(b, n, c)
        x = x + self.proj(o)
        x = x + self.mlp(self.norm2(x))
        return x


class StudentEncoder(nn.Module):
    def __init__(self, cfg: StudentConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.grid = cfg.grid
        self.stem = nn.Linear(cfg.in_pixels, cfg.dim)
        self.pos = nn.Parameter(torch.zeros(1, cfg.num_tokens, cfg.dim))

        blocks: list[nn.Module] = []
        if cfg.variant in {"hybrid", "cnn"}:
            blocks += [ConvBlock(cfg.dim, cfg.conv_kernel) for _ in range(cfg.num_conv_blocks)]
        if cfg.variant in {"hybrid", "tinyvit"}:
            blocks += [AttnBlock(cfg.dim, cfg.attn_heads, cfg.attn_mlp_ratio) for _ in range(cfg.num_attn_blocks)]
        self.blocks = nn.ModuleList(blocks)

        self.head_norm = nn.LayerNorm(cfg.dim)
        if cfg.head_hidden and cfg.head_hidden > 0:
            self.head = nn.Sequential(
                nn.Linear(cfg.dim, cfg.head_hidden), nn.GELU(),
                nn.Linear(cfg.head_hidden, cfg.seam_dim),
            )
        else:
            self.head = nn.Linear(cfg.dim, cfg.seam_dim)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        x = self.stem(patches) + self.pos
        for blk in self.blocks:
            x = blk(x, self.grid) if isinstance(blk, ConvBlock) else blk(x)
        return self.head(self.head_norm(x))


class SeamStudent(nn.Module):
    def __init__(self, cfg: StudentConfig) -> None:
        super().__init__()
        self.encoder = StudentEncoder(cfg)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        dtype = next(self.encoder.parameters()).dtype
        if images.dim() == 3:
            return self.encoder(images.to(dtype))
        b, crops, n, p = images.shape
        x = images.reshape(b * crops, n, p).to(dtype)
        seam = self.encoder(x)
        return seam.reshape(b, crops, n, seam.shape[-1])


def build_seam_student(**kwargs) -> SeamStudent:
    return SeamStudent(StudentConfig(**kwargs))


def clean_sd(sd: dict) -> dict:
    """Extract the model state_dict and strip DDP/module wrappers."""
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    out = {}
    for k, v in sd.items():
        nk = k
        for pref in ("module.", "student_model.", "model."):
            if nk.startswith(pref):
                nk = nk[len(pref):]
        out[nk] = v
    return out
