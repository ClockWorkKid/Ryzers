"""Distillation losses on the seam features (torch, cluster-only).

Primary objective: make the student's per-patch seam features match the frozen
teacher's. We combine a scale-invariant cosine term (direction) with a
normalized-MSE term (magnitude), which is more stable than raw MSE across the
2304-dim seam. An optional downstream-consistency term matches the pooled
tokens through the frozen pool+projector (what the LLM actually consumes).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class DistillLossWeights:
    cosine: float = 1.0
    norm_mse: float = 1.0
    downstream: float = 0.0   # 0 -> off; enable once pool+projector are wired


def seam_cosine_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """1 - mean per-patch cosine similarity. Shapes: [..., seam_dim]."""
    s = student.float()
    t = teacher.float()
    cos = F.cosine_similarity(s, t, dim=-1)
    return (1.0 - cos).mean()


def seam_norm_mse_loss(student: torch.Tensor, teacher: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """MSE after per-patch L2 normalization (magnitude-aware but scale-robust)."""
    s = student.float()
    t = teacher.float()
    scale = t.norm(dim=-1, keepdim=True).clamp_min(eps)
    return F.mse_loss(s / scale, t / scale)


def distill_loss(
    student_seam: torch.Tensor,
    teacher_seam: torch.Tensor,
    weights: DistillLossWeights,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Total distillation loss + a dict of scalar components for logging."""
    comps: dict[str, torch.Tensor] = {}
    total = student_seam.new_zeros((), dtype=torch.float32)

    if weights.cosine:
        comps["cosine"] = seam_cosine_loss(student_seam, teacher_seam)
        total = total + weights.cosine * comps["cosine"]
    if weights.norm_mse:
        comps["norm_mse"] = seam_norm_mse_loss(student_seam, teacher_seam)
        total = total + weights.norm_mse * comps["norm_mse"]

    logs = {k: float(v.detach()) for k, v in comps.items()}
    logs["total"] = float(total.detach())
    return total, logs


@torch.no_grad()
def fidelity_metrics(student_seam: torch.Tensor, teacher_seam: torch.Tensor) -> dict[str, float]:
    """Eval-time fidelity: mean per-patch cosine and relative L2 error."""
    s = student_seam.float()
    t = teacher_seam.float()
    cos = F.cosine_similarity(s, t, dim=-1).mean()
    rel_l2 = ((s - t).norm(dim=-1) / t.norm(dim=-1).clamp_min(1e-6)).mean()
    return {"cosine": float(cos), "rel_l2": float(rel_l2)}
