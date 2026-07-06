"""ROI-guided pre-ViT (stage-D) patch pruning for MolmoAct2.

This module is intentionally self-contained (no lerobot import) so it can be unit
tested in isolation and then wired into
``MolmoAct2VisionBackbone.encode_image`` (the patch-embedding -> resblocks seam).

Design (grounds on VLA_VisionToken_Reduction_PriorWorks_and_Proposal.md, stage D):

    x = patch_embedding(images)          # [B, N, H]
    x = add_pos_emb(x)                   # [B, N, H]
    -------- pre-ViT seam (here) --------
    scores  = score_patches(x)           # [B, N]  ROI saliency, pre-encoder
    keep    = select_topk(scores, k)     # [B, K]  indices to encode
    x_kept  = gather(x, keep)            # [B, K, H]  <-- ViT blocks run on K << N
    feats_k = resblocks(x_kept)          # per selected layer
    feats   = scatter_back(feats_k, keep, N)   # [B, N, H*L]  downstream shape kept

Selection modes:
  * ``random`` : uniform random keep-set (Phase-1 baseline; proves the FLOP win).
  * ``energy`` : parameter-free ||post-pos-emb embedding|| saliency (proposal (1)).
  * ``gate``   : learnable 2-layer MLP keep-logit, trained via a straight-through
                 top-k estimator (proposal (2)/(3), LightVLA-style).

The gate is a normal ``nn.Module`` whose parameters go into a dedicated optimizer
group (see the policy's ``get_optim_params``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PreViTPruneConfig:
    """Config for stage-D pre-ViT pruning. Defaults keep behaviour unchanged."""

    enable: bool = False
    keep_frac: float = 1.0            # fraction of patches fed to the ViT blocks
    select: str = "energy"            # "random" | "energy" | "gate"
    gate_hidden: int = 256            # hidden width of the learnable ROI gate MLP
    placeholder: str = "mask"         # scatter-back fill: "mask" (learned) | "zeros"
    min_keep: int = 1                 # never prune below this many patches

    def resolve_keep(self, num_patches: int) -> int:
        if not self.enable or self.keep_frac >= 1.0:
            return num_patches
        k = int(round(num_patches * float(self.keep_frac)))
        return max(self.min_keep, min(num_patches, k))


class ROIGate(nn.Module):
    """Tiny 2-layer MLP producing a per-patch keep-logit from patch embeddings."""

    def __init__(self, hidden_size: int, gate_hidden: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, gate_hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(gate_hidden, 1)
        # Start near-uniform (tiny output weights, zero bias) so early keep-scores
        # are close to flat, yet gradient still reaches fc1 (a pure zero-init of
        # fc2.weight would decouple fc1 since d(score)/d(h) == fc2.weight == 0).
        nn.init.normal_(self.fc2.weight, std=1e-3)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, N, H] -> [B, N]
        return self.fc2(self.act(self.fc1(x))).squeeze(-1)


def score_patches(
    x: torch.Tensor,
    mode: str,
    gate: ROIGate | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Per-patch ROI score in float32. Higher = more likely to keep. x: [B, N, H]."""
    if mode == "random":
        return torch.rand(x.shape[:2], device=x.device, generator=generator, dtype=torch.float32)
    if mode == "energy":
        # L2 norm of the post-pos-emb embedding: cheap, parameter-free saliency.
        return x.float().norm(dim=-1)
    if mode == "gate":
        if gate is None:
            raise ValueError("select='gate' requires an ROIGate instance")
        return gate(x).float()
    raise ValueError(f"unknown select mode: {mode!r}")


def select_topk(scores: torch.Tensor, keep: int) -> torch.Tensor:
    """Return sorted keep-indices [B, keep] for the top-`keep` scores per row."""
    idx = scores.topk(keep, dim=1).indices
    return idx.sort(dim=1).values  # keep original patch order for pos consistency


def gather_keep(x: torch.Tensor, keep_idx: torch.Tensor) -> torch.Tensor:
    """Gather kept patches. x: [B, N, H], keep_idx: [B, K] -> [B, K, H]."""
    H = x.shape[-1]
    return x.gather(1, keep_idx.unsqueeze(-1).expand(-1, -1, H))


def scatter_back(
    encoded_kept: torch.Tensor,
    keep_idx: torch.Tensor,
    num_patches: int,
    placeholder: torch.Tensor | str = "zeros",
) -> torch.Tensor:
    """Scatter encoded kept patches back to the full N-patch grid.

    encoded_kept: [B, K, D]; keep_idx: [B, K]; returns [B, N, D].
    Pruned positions are filled with `placeholder` (a [D] / [B,N,D] tensor, or
    "zeros"). This preserves the patch index contract that downstream pooling
    (`pooled_patches_idx`) relies on.
    """
    B, K, D = encoded_kept.shape
    if isinstance(placeholder, str):
        if placeholder != "zeros":
            raise ValueError(f"unknown placeholder: {placeholder!r}")
        full = encoded_kept.new_zeros((B, num_patches, D))
    else:
        full = placeholder.to(encoded_kept.dtype).expand(B, num_patches, D).contiguous()
    full.scatter_(1, keep_idx.unsqueeze(-1).expand(-1, -1, D), encoded_kept)
    return full


def apply_gate_ste(
    x_kept: torch.Tensor,
    scores: torch.Tensor,
    keep_idx: torch.Tensor,
) -> torch.Tensor:
    """Straight-through gate: multiply kept embeddings by sigmoid(score) with a
    forward value of 1 so magnitudes are preserved but gradients reach the gate.

    x_kept: [B, K, H], scores: [B, N], keep_idx: [B, K] -> [B, K, H].
    """
    kept_scores = scores.gather(1, keep_idx)             # [B, K]
    soft = torch.sigmoid(kept_scores)                    # differentiable weight
    ste = (1.0 - soft).detach() + soft                   # forward=1, backward=soft'
    return x_kept * ste.unsqueeze(-1).to(x_kept.dtype)
