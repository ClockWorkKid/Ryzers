"""Torch-free configuration for the teacher seam and the student encoder.

Kept import-light (stdlib only) so the FLOP profiler and unit tests run on the
laptop without torch. The teacher numbers are the exact MolmoAct2-LIBERO ViT
config verified from the model config.json and cross-checked against
``research/roi_lora_lerobot/artifacts_gen/compute_saved.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TeacherSpec:
    """Frozen MolmoAct2 SigLIP2 ViT vision tower (the distillation teacher)."""

    name: str = "MolmoAct2-LIBERO"
    # Patch grid: 378x378 input, patch 14 -> 27x27 = 729 patches/crop, no CLS.
    grid: int = 27
    num_patches: int = 729
    patch_size: int = 14
    patch_pixels: int = 14 * 14 * 3  # 588, patch_embedding input dim
    hidden_size: int = 1152
    mlp_hidden: int = 4304
    num_heads: int = 16
    head_dim: int = 72
    # Backbone runs blocks [0 .. max(vit_layers)] = 25 resblocks for vit_layers=(-3,-9).
    resblocks_run: int = 25
    # adapter.vit_layers=(-3,-9): two layers concatenated -> the seam feature dim.
    vit_layers: tuple[int, ...] = (-3, -9)

    @property
    def seam_dim(self) -> int:
        """Feature width at the distillation seam (concat of ``vit_layers``)."""
        return self.hidden_size * len(self.vit_layers)  # 2304

    @property
    def image_size(self) -> int:
        """Per-crop input side length (grid * patch_size = 378)."""
        return self.grid * self.patch_size


TEACHER_LIBERO = TeacherSpec()


@dataclass
class StudentConfig:
    """Config-driven student encoder producing the seam tensor [B, N, seam_dim].

    The student operates entirely on the 27x27 patch grid (no spatial
    down/up-sampling) so its output token count matches the teacher's 729
    patches by construction. ``variant`` selects the block topology:

    - ``hybrid`` (default): conv blocks (depthwise 3x3 + pointwise 1x1) followed
      by a few light global-attention blocks, then a linear head to ``seam_dim``.
      Best chance of matching a global-attention teacher while staying ~100x
      cheaper and reasonably hardware-friendly.
    - ``cnn``: conv blocks only (most FPGA-friendly; purely local receptive field).
    - ``tinyvit``: attention blocks only (closest to the teacher family).
    """

    variant: str = "hybrid"          # "hybrid" | "cnn" | "tinyvit"
    grid: int = 27
    in_pixels: int = 588             # per-patch raw pixels (14*14*3)
    dim: int = 256                   # internal working width
    seam_dim: int = 2304             # output width (matches TeacherSpec.seam_dim)
    num_conv_blocks: int = 4         # used by hybrid/cnn
    conv_kernel: int = 3
    num_attn_blocks: int = 3         # used by hybrid/tinyvit
    attn_heads: int = 8
    attn_mlp_ratio: float = 2.0
    head_hidden: int = 0             # 0 -> single linear head; >0 -> 2-layer head

    def __post_init__(self) -> None:
        if self.variant not in {"hybrid", "cnn", "tinyvit"}:
            raise ValueError(
                f"Unsupported variant={self.variant!r}; expected hybrid|cnn|tinyvit."
            )
        if self.dim % self.attn_heads != 0:
            raise ValueError(
                f"dim ({self.dim}) must be divisible by attn_heads ({self.attn_heads})."
            )
        if self.conv_kernel % 2 == 0:
            raise ValueError(f"conv_kernel must be odd, got {self.conv_kernel}.")

    @property
    def num_tokens(self) -> int:
        return self.grid * self.grid
