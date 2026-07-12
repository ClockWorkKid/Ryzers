"""torchdistill registry adapters (torch, cluster-only).

Importing this module registers our MolmoAct2 teacher, our student, and the seam
distillation loss with torchdistill, so an experiment is fully described by a YAML
config. This is the only glue we own; torchdistill provides the training loop,
DDP, scheduling, forward hooks, checkpointing, and the SOTA KD machinery.

The YAML references these by key:
  models.teacher_model.key: 'molmoact2_seam_teacher'
  models.student_model.key: 'vit_distill_student'
  ...criterion.sub_terms.seam.criterion.key: 'seam_distill'
"""

from __future__ import annotations

import torch
import torch.nn as nn

from torchdistill.losses.registry import register_mid_level_loss
from torchdistill.models.registry import register_model

from .losses import seam_cosine_loss, seam_norm_mse_loss
from .student import build_seam_student
from .teacher import build_seam_teacher


@register_model(key="molmoact2_seam_teacher")
def _teacher(**kwargs):
    return build_seam_teacher(**kwargs)


@register_model(key="vit_distill_student")
def _student(**kwargs):
    return build_seam_student(**kwargs)


@register_mid_level_loss(key="seam_distill")
class SeamDistillLoss(nn.Module):
    """Cosine + normalized-MSE between student and teacher seam features.

    Reads root ('.') outputs of the student/teacher modules from the io dicts
    torchdistill populates via forward hooks.
    """

    def __init__(
        self,
        student_module_path: str = ".",
        student_module_io: str = "output",
        teacher_module_path: str = ".",
        teacher_module_io: str = "output",
        cosine_weight: float = 1.0,
        norm_mse_weight: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.student_module_path = student_module_path
        self.student_module_io = student_module_io
        self.teacher_module_path = teacher_module_path
        self.teacher_module_io = teacher_module_io
        self.cosine_weight = cosine_weight
        self.norm_mse_weight = norm_mse_weight

    def forward(self, student_io_dict, teacher_io_dict, *args, **kwargs) -> torch.Tensor:
        s = student_io_dict[self.student_module_path][self.student_module_io]
        t = teacher_io_dict[self.teacher_module_path][self.teacher_module_io].detach()
        loss = s.new_zeros((), dtype=torch.float32)
        if self.cosine_weight:
            loss = loss + self.cosine_weight * seam_cosine_loss(s, t)
        if self.norm_mse_weight:
            loss = loss + self.norm_mse_weight * seam_norm_mse_loss(s, t)
        return loss
