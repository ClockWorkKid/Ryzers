"""Dependency-light DROID image stream for the Run D joint finetune.

Provides raw DROID frames (pre-extracted JPEGs under DROID_ROOT) patchified into
the exact tensor the MolmoAct2 ViT consumes, so a seam-retention distillation
term (student ViT vs frozen teacher ViT) can be added each training step. This
mirrors the run B "hybrid_droid" mixing that best retained closed-loop
performance, but carries it into the joint ViT+LLM finetune.

Kept free of the vit_distill package (only torch + PIL + transformers) so it can
be bind-mounted into the lerobot eval SIF. The patchify + normalization exactly
reproduce research/vit_distill/distill/{patchify,teacher}.py.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

_PIXEL_KEYS = ("images", "pixel_values", "input_pixels", "image_patches")


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


def seam_cosine_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """1 - mean per-patch cosine similarity over the seam dim."""
    s = student.float()
    t = teacher.float()
    cos = F.cosine_similarity(s, t, dim=-1)
    return (1.0 - cos).mean()


def seam_norm_mse_loss(student: torch.Tensor, teacher: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """MSE after per-patch L2 normalization (magnitude-aware, scale-robust)."""
    s = student.float()
    t = teacher.float()
    scale = t.norm(dim=-1, keepdim=True).clamp_min(eps)
    return F.mse_loss(s / scale, t / scale)


class _FramePatchifier:
    """frame (uint8 HWC) -> normalized patches [num_crops, N, in_pixels]."""

    def __init__(self, checkpoint_path: str = "allenai/MolmoAct2-LIBERO", revision: str | None = None):
        from transformers import AutoProcessor

        self._processor = AutoProcessor.from_pretrained(
            checkpoint_path, revision=revision, trust_remote_code=True
        )

    def __call__(self, frame_u8: torch.Tensor) -> torch.Tensor:
        from PIL import Image

        pil = Image.fromarray(frame_u8.cpu().numpy().astype(np.uint8))
        img_proc = getattr(self._processor, "image_processor", None)
        if img_proc is not None and hasattr(img_proc, "preprocess"):
            out = img_proc.preprocess(images=[pil], return_tensors="pt")
        else:
            out = self._processor(images=[pil], text="describe the image.", return_tensors="pt")
        pixels = None
        for k in _PIXEL_KEYS:
            if k in out:
                pixels = out[k]
                break
        if pixels is None:
            for v in out.values():
                if torch.is_tensor(v) and v.dim() >= 3 and torch.is_floating_point(v):
                    pixels = v
                    break
        if pixels is None:
            raise RuntimeError(f"Could not locate pixel patches in processor output keys={list(out)}")
        pixels = pixels.squeeze(0)
        if pixels.dim() == 2:
            pixels = pixels.unsqueeze(0)
        return normalize_patches(pixels, torch.float32)


class DroidPatchDataset(Dataset):
    """DROID JPEG frames -> teacher-ready normalized patch tensors."""

    def __init__(self, root: str, checkpoint_path: str = "allenai/MolmoAct2-LIBERO",
                 max_frames: int | None = None, seed: int = 0):
        exts = ("*.jpg", "*.jpeg", "*.png")
        files: list[str] = []
        for e in exts:
            files += glob.glob(os.path.join(root, "**", e), recursive=True)
        files = sorted(files)
        if not files:
            raise RuntimeError(f"No DROID image frames (jpg/png) under {root}")
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(files))
        files = [files[i] for i in order]
        if max_frames is not None:
            files = files[:max_frames]
        self.files = files
        self.checkpoint_path = checkpoint_path
        self._patchify = None  # lazily built per worker

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        from PIL import Image

        if self._patchify is None:
            self._patchify = _FramePatchifier(self.checkpoint_path)
        img = Image.open(self.files[idx]).convert("RGB")
        arr = torch.from_numpy(np.array(img, dtype=np.uint8))  # HWC uint8
        return self._patchify(arr)  # [num_crops, N, in_pixels]


def _collate(batch: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack(batch, dim=0)  # [B, num_crops, N, in_pixels]


class DroidBatcher:
    """Infinite DROID patch-batch source; re-iterates the loader when exhausted."""

    def __init__(self, root: str, batch_size: int = 4, num_workers: int = 2,
                 checkpoint_path: str = "allenai/MolmoAct2-LIBERO", max_frames: int | None = None):
        try:
            torch.multiprocessing.set_sharing_strategy("file_system")
        except Exception:
            pass
        self.ds = DroidPatchDataset(root, checkpoint_path=checkpoint_path, max_frames=max_frames)
        self._loader = DataLoader(
            self.ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
            collate_fn=_collate, drop_last=True, persistent_workers=(num_workers > 0),
            pin_memory=False,
        )
        self._it = iter(self._loader)

    def next(self) -> torch.Tensor:
        try:
            return next(self._it)
        except StopIteration:
            self._it = iter(self._loader)
            return next(self._it)
