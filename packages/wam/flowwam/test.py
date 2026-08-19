# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Environment sign-of-life for the FlowWAM model layer on Strix Halo (gfx1151).

Runs inside the built image with NO model weights. Proves the container has (1) a working
ROCm torch on the iGPU, (2) the FlowWAM/DiffSynth pipeline + dual-stream flow modules import
cleanly, (3) SAPIEN + the RAFT/optical-flow + I/O deps import, and (4) DiffSynth attention will
use the torch-SDPA fallback (no CUDA flash-attn present) -- all before we pull the multi-GB
Wan2.2-TI2V-5B base + FlowWAM checkpoint. Exits non-zero on any failure so `ryzers run` / CI
catches a broken image early.
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
    a = torch.randn(512, 512, device="cuda")
    b = torch.randn(512, 512, device="cuda")
    print(f"matmul ok        : sum={(a @ b).sum().item():.3f}")

    # FlowWAM / DiffSynth Wan dual-stream world model + core runtime deps.
    from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig  # noqa: F401
    from diffsynth.models.wan_video_dit_dual_stream import init_flow_stream      # noqa: F401
    from diffsynth.pipelines.wan_video_dual_stream import model_fn_wan_video_dual_stream  # noqa: F401
    from diffsynth.models import wan_video_dit
    import transformers, einops, safetensors, sentencepiece                      # noqa: F401
    import cv2, h5py, imageio, accelerate                                        # noqa: F401
    import sapien                                                                # noqa: F401

    # Confirm we are on the torch-SDPA fallback path (no CUDA flash-attn / sage-attn on ROCm).
    fa3 = getattr(wan_video_dit, "FLASH_ATTN_3_AVAILABLE", False)
    fa2 = getattr(wan_video_dit, "FLASH_ATTN_2_AVAILABLE", False)
    sage = getattr(wan_video_dit, "SAGE_ATTN_AVAILABLE", False)
    print(f"attention backend: flash3={fa3} flash2={fa2} sage={sage} -> "
          f"{'SDPA fallback (ROCm)' if not (fa3 or fa2 or sage) else 'accelerated'}")

    # Exercise the SDPA path once so we know Wan-DiT attention runs on the iGPU.
    with torch.no_grad():
        q = torch.randn(1, 128, 16 * 64, device="cuda", dtype=torch.bfloat16)
        out = wan_video_dit.flash_attention(q, q, q, num_heads=16)
    assert out.shape == q.shape and torch.isfinite(out).all(), "flash_attention SDPA path broken"
    print(f"wan attention ok : out={tuple(out.shape)} (bf16 SDPA)")

    print(f"transformers     : {transformers.__version__}  sapien: {sapien.__version__}")
    print("deps import ok   : diffsynth(Wan dual-stream), sapien, cv2, h5py, imageio, accelerate")
    print("PASS: FlowWAM ROCm model-layer env OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
