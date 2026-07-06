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
                 Position-dominated -> near-fixed spatial keep-pattern across frames.
  * ``content``: parameter-free ||pre-pos-emb patch projection|| saliency. Same norm
                 as ``energy`` but scored BEFORE the positional embedding is added, so
                 the keep-set is image-adaptive (tracks textured/foreground patches)
                 instead of position-locked. ViT still consumes the post-pos-emb
                 embeddings of the kept patches.
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
    select: str = "energy"            # "random"|"energy"|"content"|"gate"|"actionattn"|"gate_distill"|"gate_predict"
    gate_hidden: int = 256            # hidden width of the learnable ROI gate MLP
    placeholder: str = "mask"         # scatter-back fill: "mask" (learned) | "zeros"
    min_keep: int = 1                 # never prune below this many patches
    teacher_layers: tuple = (9, 20, 21)  # action-expert layers for the attention teacher
    teacher_debias: bool = True       # subtract per-layer batch-mean (positional sink) template
    distill_weight: float = 1.0       # weight of the gate<-teacher distillation loss (variants B/C)
    gate_seam: int = 0                 # #ViT blocks to run on ALL patches before the gate scores/prunes
                                       # (0 = pre-ViT/raw projection; >0 = score semantic mid-ViT feats)
    group_drop: bool = False           # prune WHOLE 2x2 pooling groups so the pooled/LLM token count
                                       # drops ~linearly with keep_frac (real backbone compute saving),
                                       # instead of scatter-back (which keeps the token count fixed)

    @property
    def uses_action_teacher(self) -> bool:
        return self.select in ("actionattn", "gate_distill", "gate_predict")

    @property
    def uses_gate(self) -> bool:
        return self.select in ("gate", "gate_distill", "gate_predict")

    def resolve_keep(self, num_patches: int) -> int:
        if not self.enable or self.keep_frac >= 1.0:
            return num_patches
        k = int(round(num_patches * float(self.keep_frac)))
        return max(self.min_keep, min(num_patches, k))


class ROIGate(nn.Module):
    """Tiny per-patch keep-logit MLP, optionally conditioned on a task/instruction
    vector (FiLM-style additive shift broadcast over patches).

    A parameter-free content gate cannot recover the action-attention teacher ROI:
    the teacher is task-conditioned (which object to grasp depends on the language
    instruction), so a per-patch scorer that only sees patch appearance has no way
    to know WHICH object matters. When ``task_dim`` is given the gate projects a
    pooled instruction embedding into the hidden and adds it to every patch, so the
    keep-logit becomes a function of (patch feature, instruction).
    """

    def __init__(self, hidden_size: int, gate_hidden: int = 256, task_dim: int | None = None):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, gate_hidden)
        self.task_proj = nn.Linear(task_dim, gate_hidden) if task_dim else None
        self.act = nn.GELU()
        self.fc2 = nn.Linear(gate_hidden, 1)
        # Start near-uniform (tiny output weights, zero bias) so early keep-scores
        # are close to flat, yet gradient still reaches fc1 (a pure zero-init of
        # fc2.weight would decouple fc1 since d(score)/d(h) == fc2.weight == 0).
        nn.init.normal_(self.fc2.weight, std=1e-3)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor, task: torch.Tensor | None = None) -> torch.Tensor:
        # x: [B, N, H]; task: [B, task_dim] -> logits [B, N]
        h = self.fc1(x)
        if self.task_proj is not None and task is not None:
            h = h + self.task_proj(task.to(h.dtype)).unsqueeze(1)
        return self.fc2(self.act(h)).squeeze(-1)


class ROIGateXAttn(nn.Module):
    """Task-conditioned keep-logit gate with a patch->instruction cross-attention.

    An additive/FiLM task term is a per-image constant and cannot change WHICH patch
    ranks highest (verified empirically: it adds nothing over a position-only gate).
    To make the instruction actually select patches we need a patch x instruction
    INTERACTION: each patch forms a query and attends over the instruction token
    embeddings (keys/values), so a patch that matches words in the instruction pulls
    in relevant context and can be scored up. The keep-logit is an MLP on
    [patch_feature ; attended_instruction_context].

    task = (tokens, mask): tokens [B, T, task_dim] instruction-token embeddings,
    mask [B, T] bool (True = real token). When task is None the gate degrades to a
    patch-only scorer (context set to zero) so inference without an instruction still
    works.
    """

    def __init__(self, hidden_size: int, task_dim: int, gate_hidden: int = 256):
        super().__init__()
        self.scale = float(gate_hidden) ** -0.5
        self.q = nn.Linear(hidden_size, gate_hidden)
        self.k = nn.Linear(task_dim, gate_hidden)
        self.v = nn.Linear(task_dim, gate_hidden)
        self.patch_proj = nn.Linear(hidden_size, gate_hidden)
        self.fc = nn.Linear(2 * gate_hidden, gate_hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(gate_hidden, 1)
        nn.init.normal_(self.fc2.weight, std=1e-3)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor, task=None) -> torch.Tensor:
        # x: [B, N, H] -> logits [B, N]
        ctx = self.patch_proj.weight.new_zeros(x.shape[0], x.shape[1], self.q.out_features)
        if task is not None:
            tokens, mask = task
            tokens = tokens.to(x.dtype)
            q = self.q(x) * self.scale                     # [B, N, g]
            k = self.k(tokens)                             # [B, T, g]
            v = self.v(tokens)                             # [B, T, g]
            att = torch.matmul(q, k.transpose(1, 2))       # [B, N, T]
            if mask is not None:
                att = att.masked_fill(~mask[:, None, :].to(att.device), float("-inf"))
            att = torch.softmax(att.float(), dim=-1).to(v.dtype)
            att = torch.nan_to_num(att)                    # rows with no valid token -> 0
            ctx = torch.matmul(att, v)                     # [B, N, g]
        h = self.act(self.fc(torch.cat([self.patch_proj(x), ctx], dim=-1)))
        return self.fc2(h).squeeze(-1)


def score_patches(
    x: torch.Tensor,
    mode: str,
    gate: ROIGate | None = None,
    generator: torch.Generator | None = None,
    task: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-patch ROI score in float32. Higher = more likely to keep. x: [B, N, H]."""
    if mode == "random":
        return torch.rand(x.shape[:2], device=x.device, generator=generator, dtype=torch.float32)
    if mode in ("energy", "content"):
        # L2 norm of the patch embedding: cheap, parameter-free saliency. The caller
        # decides WHICH embedding is passed in: "energy" scores the post-pos-emb
        # tensor (position-dominated, near-fixed spatial mask), while "content"
        # scores the pre-pos-emb patch projection (image-adaptive ROI, no position
        # bias). The math is identical; only the input tensor differs.
        return x.float().norm(dim=-1)
    if mode == "gate":
        if gate is None:
            raise ValueError("select='gate' requires an ROIGate instance")
        return gate(x, task=task).float()
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
