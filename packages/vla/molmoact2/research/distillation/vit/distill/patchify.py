"""Turn a raw frame into the teacher's patchified pixel tensor (torch, cluster-only).

Both teacher and student must consume the *identical* normalized patch tensor.
We prefer the released MolmoAct2 processor (exact resize/crop/patch layout the ViT
was trained on); a deterministic fallback is provided so the pipeline is testable
even if the processor's multimodal call signature drifts.

M1 verification (rule 2): confirm on-cluster that `processor_patchify` returns
[num_crops, 729, 588] matching the teacher, and that `normalize_patches` reproduces
`MolmoAct2VisionBackbone.forward`'s pixel scaling exactly (compare a live seam).
"""

from __future__ import annotations

import numpy as np
import torch

from .config import TEACHER_LIBERO
from .teacher import normalize_patches

# Candidate keys the processor may use for the patchified pixels; autodetected.
_PIXEL_KEYS = ("images", "pixel_values", "input_pixels", "image_patches")


class FramePatchifier:
    """frame (uint8 HWC) -> normalized patches [num_crops, N, in_pixels] float32."""

    def __init__(self, mode: str = "processor", checkpoint_path: str = "allenai/MolmoAct2-LIBERO",
                 revision: str | None = None, prompt: str = "describe the image.") -> None:
        self.mode = mode
        self.prompt = prompt
        self._processor = None
        if mode == "processor":
            from .teacher import load_image_processor

            self._processor = load_image_processor(checkpoint_path, revision)

    def _processor_patchify(self, frame_u8: torch.Tensor) -> torch.Tensor:
        from PIL import Image

        pil = Image.fromarray(frame_u8.cpu().numpy().astype(np.uint8))
        # Fast path: call the image processor directly to skip text/video
        # tokenization. This yields the same `pixel_values` as the full
        # multimodal call but avoids the prompt-assembly overhead that
        # dominated data-loading time in the M2 smoke run.
        img_proc = getattr(self._processor, "image_processor", None)
        if img_proc is not None and hasattr(img_proc, "preprocess"):
            out = img_proc.preprocess(images=[pil], return_tensors="pt")
        else:
            out = self._processor(images=[pil], text=self.prompt, return_tensors="pt")
        pixels = None
        for k in _PIXEL_KEYS:
            if k in out:
                pixels = out[k]
                break
        if pixels is None:  # last resort: first 3D+ float tensor
            for v in out.values():
                if torch.is_tensor(v) and v.dim() >= 3 and torch.is_floating_point(v):
                    pixels = v
                    break
        if pixels is None:
            raise RuntimeError(f"Could not locate pixel patches in processor output keys={list(out)}")
        pixels = pixels.squeeze(0)  # drop batch dim -> [num_crops, N, in_pixels]
        if pixels.dim() == 2:  # [N, in_pixels] -> add crop dim
            pixels = pixels.unsqueeze(0)
        return pixels

    def _simple_patchify(self, frame_u8: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F

        t = TEACHER_LIBERO
        side = t.image_size  # e.g. 378
        x = frame_u8.permute(2, 0, 1).unsqueeze(0).float()
        x = F.interpolate(x, size=(side, side), mode="bilinear", align_corners=False)
        x = x.squeeze(0).to(torch.uint8)  # [3, side, side]
        g = side // t.patch_size  # patches per side (e.g. 27)
        p = t.patch_size
        # [3, g, p, g, p] -> [g*g, p*p*3]
        patches = x.reshape(3, g, p, g, p).permute(1, 3, 2, 4, 0).reshape(g * g, p * p * 3)
        return patches.unsqueeze(0)  # [1, N, in_pixels]

    def __call__(self, frame_u8: torch.Tensor) -> torch.Tensor:
        raw = self._processor_patchify(frame_u8) if self.mode == "processor" else self._simple_patchify(frame_u8)
        return normalize_patches(raw, torch.float32)
