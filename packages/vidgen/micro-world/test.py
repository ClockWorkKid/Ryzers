# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Environment sign-of-life for the Micro-World video-gen image on Strix Halo (gfx1151).

Runs inside the built image with NO model weights. Proves, independently (rule 2):
  1. a working ROCm torch on the iGPU (hip build + visible device),
  2. the curated dep stack imports (diffusers==0.34.0, transformers, ...),
  3. every Micro-World module imports (all model classes, all 4 pipelines, utils, schedulers),
  4. the ROCm gfx1151 SDPA fallback in wan_transformer3d.flash_attention() works on-device (this is
     the CLIP-encoder path the I2V/I2W examples hit — no flash-attn wheel needed), and that the
     DiT attention() dispatcher also runs, and
  5. parse_action_list() decodes the documented action-string format.
Exits non-zero on any failure so `ryzers run` / CI catches a broken image early.
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
    print(f"diffusers        : {diffusers.__version__}")
    print(f"transformers     : {transformers.__version__}")

    # (3) every Micro-World module must import independently.
    from microworld.models import (
        AutoencoderKLWan, WanT5EncoderModel, WanTransformer3DModel, WanSelfAttention,
        WanActionControlNetModel, WanActionAdaLNModel, CLIPModel,
    )  # noqa: F401
    from microworld.models.cache_utils import TeaCache, get_teacache_coefficients  # noqa: F401
    from microworld.models import wan_transformer3d
    from microworld.pipeline import (
        WanFunPipeline, WanFunInpaintPipeline, WanActionT2WPipeline, WanActionI2WPipeline,
        WanPipeline, WanI2VPipeline,
    )  # noqa: F401
    from microworld.utils.utils import (
        filter_kwargs, save_videos_grid, parse_action_list,
        get_image_to_video_latent, replace_parameters_by_name,
    )  # noqa: F401
    from microworld.utils.fm_solvers import FlowDPMSolverMultistepScheduler  # noqa: F401
    from microworld.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler  # noqa: F401
    from microworld.utils.lora_utils import merge_lora, unmerge_lora  # noqa: F401
    print("imports ok       : models.{vae,t5,dit,controlnet,adaln,clip}, cache_utils, "
          "pipeline.{t2w,i2w,fun,fun_inpaint}, utils, fm_solvers, lora_utils")
    print(f"flash_attn avail : FA2={wan_transformer3d.FLASH_ATTN_2_AVAILABLE} "
          f"FA3={wan_transformer3d.FLASH_ATTN_3_AVAILABLE} (expected False/False on gfx1151)")

    # (4) On-device attention: the SDPA fallback (CLIP AttentionPool path) + the DiT dispatcher.
    dev = "cuda"
    B, L, N, C = 1, 16, 8, 64  # (batch, seq, heads, head_dim)
    q = torch.randn(B, L, N, C, device=dev, dtype=torch.bfloat16)
    k = torch.randn(B, L, N, C, device=dev, dtype=torch.bfloat16)
    v = torch.randn(B, L, N, C, device=dev, dtype=torch.bfloat16)
    with torch.no_grad():
        # flash_attention() is what wan_image_encoder.AttentionPool calls directly.
        out_fa = wan_transformer3d.flash_attention(q, k, v, version=2)
        # attention() is the DiT dispatcher (default VIDEOX_ATTENTION_TYPE=FLASH_ATTENTION -> SDPA).
        out_at = wan_transformer3d.attention(q, k, v)
    assert tuple(out_fa.shape) == (B, L, N, C), f"flash_attention bad shape {tuple(out_fa.shape)}"
    assert tuple(out_at.shape) == (B, L, N, C), f"attention bad shape {tuple(out_at.shape)}"
    print(f"attention ok     : flash_attention(SDPA fallback)={tuple(out_fa.shape)} "
          f"dispatcher={tuple(out_at.shape)}")

    # (5) action-string decoding (documented format).
    kb, ms = parse_action_list([[20, "1 0 0 0 0 0 0 0 0"], [40, "0 1 0 0 0 0 0 0 5"], "30 60"])
    print(f"parse_action_list: keyboard={tuple(kb.shape)} mouse={tuple(ms.shape)}")

    print("PASS: Micro-World ROCm env OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
