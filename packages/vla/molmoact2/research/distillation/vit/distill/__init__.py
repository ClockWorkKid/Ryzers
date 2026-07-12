"""MolmoAct2 vision-encoder distillation (research-only).

Distill the frozen MolmoAct2 SigLIP2 ViT into a ~100x-lighter student that
reproduces the concatenated ``vit_layers`` seam features ([B, 729, 2304]) that
feed the (frozen) 2x2 attention pooling + projector. See README.md.

Distillation is driven by torchdistill (config-driven KD engine); our code is
thin adapters registered via ``distill.register`` and selected from a YAML config
under ``configs/``. See README.md.

Torch-free modules (import safely without torch): ``config``, ``flops``.
Torch modules (require torch on the cluster image): ``student``, ``teacher``,
``losses``, ``data``, ``patchify``, ``register``.
"""

from .config import StudentConfig, TeacherSpec, TEACHER_LIBERO

__all__ = ["StudentConfig", "TeacherSpec", "TEACHER_LIBERO"]
