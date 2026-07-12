"""Frozen MolmoAct2 teacher: seam-feature module (torch, cluster-only).

The distillation seam is the concatenated ``vit_layers`` tensor that
``MolmoAct2VisionBackbone.encode_image`` returns, shape
``[B, num_crops, 729, 2304]`` -- exactly the input to the frozen 2x2 attention
pool. We reuse the released checkpoint's own vision backbone (loaded via
``trust_remote_code``); nothing from the failed ROI run is used.

Input contract (shared verbatim with the student): normalized, patchified
pixels ``[B, num_crops, 729, 588]`` (normalization = the exact ops from
``MolmoAct2VisionBackbone.forward``). ``SeamTeacher`` and the student both consume
this identical tensor so the distillation compares like-for-like.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def normalize_patches(images: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Replicate MolmoAct2VisionBackbone.forward pixel normalization."""
    if images.dtype == torch.uint8:
        images = images.to(torch.float32) / 255.0
        images = images * 2.0 - 1.0
    elif torch.is_floating_point(images):
        images = torch.round(((images.to(torch.float32) + 1.0) * 0.5) * 255.0)
        images = torch.clamp(images, 0.0, 255.0) / 255.0
        images = images * 2.0 - 1.0
    return images.to(dtype=dtype)


class SeamTeacher(nn.Module):
    """Frozen wrapper: patchified pixels -> teacher seam features.

    forward(images): images [B, num_crops, N, in_pixels] -> [B, num_crops, N, seam_dim].
    Kept as an ``nn.Module`` whose *root* output is the seam, so torchdistill can
    grab it via ``teacher_module_path='.'``.
    """

    def __init__(self, backbone: nn.Module, dtype: torch.dtype = torch.bfloat16,
                 already_normalized: bool = False) -> None:
        super().__init__()
        self.backbone = backbone
        self.dtype = dtype
        self.already_normalized = already_normalized
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = images if self.already_normalized else normalize_patches(images, self.dtype)
        return self.backbone.encode_image(x.to(self.dtype))

    def train(self, mode: bool = True):  # keep frozen backbone in eval always
        super().train(mode)
        self.backbone.eval()
        return self


def load_vision_backbone(
    checkpoint_path: str = "allenai/MolmoAct2-LIBERO",
    revision: str | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> nn.Module:
    """Load the full model (trust_remote_code) and return only the vision backbone."""
    from transformers import AutoModelForImageTextToText

    model = AutoModelForImageTextToText.from_pretrained(
        checkpoint_path, revision=revision, trust_remote_code=True, torch_dtype=dtype
    )
    backbone = model.vision_backbone
    del model  # free the LLM / action expert; we only need the ViT seam
    return backbone


def build_seam_teacher(
    checkpoint_path: str = "allenai/MolmoAct2-LIBERO",
    revision: str | None = None,
    model_dtype: str = "bfloat16",
    already_normalized: bool = False,
) -> SeamTeacher:
    """Registry entrypoint for torchdistill (`get_model` -> this)."""
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[model_dtype]
    backbone = load_vision_backbone(checkpoint_path, revision, dtype)
    return SeamTeacher(backbone, dtype=dtype, already_normalized=already_normalized)


def load_image_processor(
    checkpoint_path: str = "allenai/MolmoAct2-LIBERO", revision: str | None = None
):
    """MolmoAct2 processor that patchifies raw frames into [num_crops, 729, 588]."""
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(
        checkpoint_path, revision=revision, trust_remote_code=True
    )
