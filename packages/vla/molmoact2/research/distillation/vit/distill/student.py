"""Student vision encoder (torch). Requires torch; import only on the cluster.

Produces the distillation seam tensor of shape ``[B, N, seam_dim]`` (default
[B, 729, 2304]) so it drops in where the teacher's concatenated ``vit_layers``
features feed the frozen 2x2 attention pool. The student never changes spatial
resolution (operates on the 27x27 = 729 grid throughout) so its token count
matches the teacher by construction.

The block topology mirrors ``distill.flops`` exactly so the analytic ~100x FLOP
budget is faithful to the built module.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import StudentConfig


class ConvBlock(nn.Module):
    """Depthwise KxK + pointwise 1x1, pre-norm with residual (ConvNeXt-ish)."""

    def __init__(self, dim: int, kernel: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dw = nn.Conv2d(dim, dim, kernel, padding=kernel // 2, groups=dim)
        self.pw = nn.Linear(dim, dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, grid: int) -> torch.Tensor:
        # x: [B, N, C]
        b, n, c = x.shape
        h = self.norm(x)
        h = h.transpose(1, 2).reshape(b, c, grid, grid)   # [B, C, H, W]
        h = self.dw(h)
        h = h.reshape(b, c, n).transpose(1, 2)            # [B, N, C]
        h = self.act(self.pw(h))
        return x + h


class AttnBlock(nn.Module):
    """Pre-norm multi-head self-attention + MLP, both residual."""

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
        o = F.scaled_dot_product_attention(q, k, v)       # [B, heads, N, head_dim]
        o = o.transpose(1, 2).reshape(b, n, c)
        x = x + self.proj(o)
        x = x + self.mlp(self.norm2(x))
        return x


class StudentEncoder(nn.Module):
    """Config-driven ~100x-lighter student for the MolmoAct2 ViT seam."""

    def __init__(self, cfg: StudentConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.grid = cfg.grid
        self.stem = nn.Linear(cfg.in_pixels, cfg.dim)
        self.pos = nn.Parameter(torch.zeros(1, cfg.num_tokens, cfg.dim))
        nn.init.trunc_normal_(self.pos, std=0.02)

        blocks: list[nn.Module] = []
        if cfg.variant in {"hybrid", "cnn"}:
            blocks += [ConvBlock(cfg.dim, cfg.conv_kernel) for _ in range(cfg.num_conv_blocks)]
        if cfg.variant in {"hybrid", "tinyvit"}:
            blocks += [
                AttnBlock(cfg.dim, cfg.attn_heads, cfg.attn_mlp_ratio)
                for _ in range(cfg.num_attn_blocks)
            ]
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
        """patches: [B, N, in_pixels] -> seam: [B, N, seam_dim]."""
        if patches.dim() != 3 or patches.shape[1] != self.cfg.num_tokens:
            raise ValueError(
                f"expected [B, {self.cfg.num_tokens}, {self.cfg.in_pixels}], "
                f"got {tuple(patches.shape)}"
            )
        x = self.stem(patches) + self.pos
        for blk in self.blocks:
            x = blk(x, self.grid) if isinstance(blk, ConvBlock) else blk(x)
        return self.head(self.head_norm(x))

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class SeamStudent(nn.Module):
    """Crop-folding wrapper so the student matches the teacher's output shape.

    forward(images): [B, num_crops, N, in_pixels] -> [B, num_crops, N, seam_dim],
    mirroring ``SeamTeacher`` so torchdistill can compare root outputs directly.
    Also accepts a bare [B, N, in_pixels] (num_crops implied = 1).
    """

    def __init__(self, cfg: StudentConfig) -> None:
        super().__init__()
        self.encoder = StudentEncoder(cfg)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # cast input to the encoder's parameter dtype (bf16/fp32) to avoid mat dtype
        # mismatches; the seam loss upcasts to fp32 internally.
        dtype = next(self.encoder.parameters()).dtype
        if images.dim() == 3:
            return self.encoder(images.to(dtype))
        b, crops, n, p = images.shape
        x = images.reshape(b * crops, n, p).to(dtype)
        seam = self.encoder(x)
        return seam.reshape(b, crops, n, seam.shape[-1])

    def num_params(self) -> int:
        return self.encoder.num_params()


def build_student(cfg: StudentConfig | None = None) -> StudentEncoder:
    return StudentEncoder(cfg or StudentConfig())


def build_seam_student(**kwargs) -> SeamStudent:
    """Registry entrypoint for torchdistill; kwargs map to StudentConfig fields."""
    return SeamStudent(StudentConfig(**kwargs))
