# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Environment sign-of-life for the Latte video-gen image on Strix Halo (gfx1151).

Runs inside the built image with NO model weights: proves (1) a working ROCm torch on the
iGPU, (2) Latte's package + runtime deps import cleanly (both generation paths), and (3) a
random-input forward of the class-conditional DiT (Latte-S/2, math + flash/SDPA) runs on the
device. Exits non-zero on any failure so `ryzers run` / CI catches a broken image early.
"""
import sys


def main() -> int:
    import torch

    print(f"torch            : {torch.__version__}")
    print(f"torch.version.hip: {torch.version.hip}")
    if not torch.version.hip:
        print("FAIL: torch is not a ROCm build.", file=sys.stderr)
        return 1
    if not torch.cuda.is_available():
        print("FAIL: no ROCm device visible. Check --device=/dev/kfd, /dev/dri.", file=sys.stderr)
        return 1
    print(f"device[0]        : {torch.cuda.get_device_name(0)}")

    import diffusers
    import transformers
    import einops          # noqa: F401
    import timm            # noqa: F401
    import omegaconf       # noqa: F401
    import imageio         # noqa: F401
    from diffusers.models import AutoencoderKL  # noqa: F401
    print(f"diffusers        : {diffusers.__version__}")
    print(f"transformers     : {transformers.__version__}")

    # Both Latte generation paths must import.
    from models import get_models                     # noqa: F401
    from models.latte import Latte_models
    from models.latte_img import LatteIMG_models      # noqa: F401
    from models.latte_t2v import LatteT2V             # noqa: F401
    from diffusion import create_diffusion            # noqa: F401
    from sample.pipeline_latte import LattePipeline   # noqa: F401
    print("imports ok       : models.{latte,latte_img,latte_t2v}, diffusion, pipeline_latte")

    # Random-input forward of the smallest class-conditional DiT on the iGPU (math + flash/SDPA).
    dev = "cuda"
    B, F, C, H, W = 1, 4, 4, 32, 32  # (b, frames, latent_ch, h, w)
    for mode in ("math", "flash"):
        model = Latte_models["Latte-S/2"](
            input_size=32, num_classes=101, num_frames=F, learn_sigma=True, extras=2,
        ).to(dev).eval()
        model.blocks.apply(lambda m: setattr(m, "attention_mode", mode) if hasattr(m, "attention_mode") else None)
        # attention_mode lives on Attention submodules; set directly.
        for mod in model.modules():
            if hasattr(mod, "attention_mode"):
                mod.attention_mode = mode
        x = torch.randn(B, F, C, H, W, device=dev)
        t = torch.randint(0, 1000, (B,), device=dev)
        y = torch.randint(0, 101, (B,), device=dev)
        with torch.no_grad():
            out = model(x, t, y=y)
        assert out.shape[0] == B and out.shape[1] == F, f"bad output shape {tuple(out.shape)}"
        print(f"DiT forward ok   : mode={mode:5s} out={tuple(out.shape)}")
        del model
        torch.cuda.empty_cache()

    print("PASS: Latte ROCm env OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
