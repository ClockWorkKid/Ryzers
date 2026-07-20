# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Environment sign-of-life for the AVDC video-diffusion policy on Strix Halo (gfx1151).

Runs inside the built image with NO model weights and NO simulator: proves (1) a working ROCm
torch on the iGPU, (2) AVDC's model-path deps import cleanly (flowdiffusion + vendored guided-
diffusion 3D UNet + transformers CLIP), (3) the Meta-World 3D-UNet runs a random-input forward on
device (rule 2), and (4) the full GoalGaussianDiffusion DDIM sampling loop produces a predicted
video tensor of the expected shape. The CLIP text encoder is bypassed (random task tokens) so the
smoke needs no weight download; the real CLIP ViT-B/32 is exercised in the phase-2 open-loop demo.
Exits non-zero on any failure so `ryzers run` / CI catches a broken image early.
"""
import sys

# AVDC Meta-World config (inference_utils.get_video_model / unet.UnetMW).
SAMPLE_PER_SEQ = 8            # 1 conditioning frame + 7 predicted future frames
TARGET_SIZE = (128, 128)
CHANNELS = 3 * (SAMPLE_PER_SEQ - 1)   # 21 = 7 future RGB frames flattened
TASK_TOKENS = 512            # CLIP ViT-B/32 text hidden dim == UNet task_token_channels


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

    import transformers                                   # noqa: F401
    import einops                                         # noqa: F401
    from flowdiffusion.unet import UnetMW
    from flowdiffusion.goal_diffusion import GoalGaussianDiffusion
    print(f"transformers     : {transformers.__version__}")
    print("imports ok       : flowdiffusion.{unet,goal_diffusion} (+ vendored guided_diffusion 3D UNet)")

    dev = "cuda"
    B = 1
    f = SAMPLE_PER_SEQ - 1   # 7

    # (1) Meta-World 3D UNet forward: cat([noisy 7-frame target (21ch), cond frame (3ch)]) -> 21ch.
    unet = UnetMW().to(dev).eval()
    x_full = torch.randn(B, CHANNELS + 3, TARGET_SIZE[0], TARGET_SIZE[1], device=dev)  # 24ch
    t = torch.randint(0, 100, (B,), device=dev)
    task_embed = torch.randn(B, 77, TASK_TOKENS, device=dev)   # CLIP token stream (stubbed)
    with torch.no_grad():
        out = unet(x_full, t, task_embed)
    assert tuple(out.shape) == (B, CHANNELS, *TARGET_SIZE), out.shape
    print(f"UnetMW (3D) ok   : in={tuple(x_full.shape)} -> out={tuple(out.shape)}")

    # (2) Full DDIM sampling loop (2 steps, random weights) -> predicted video latent.
    diffusion = GoalGaussianDiffusion(
        channels=CHANNELS, model=unet, image_size=TARGET_SIZE,
        timesteps=100, sampling_timesteps=2, loss_type="l2", objective="pred_v",
        beta_schedule="cosine", min_snr_loss_weight=True).to(dev).eval()
    x_cond = torch.randn(B, 3, *TARGET_SIZE, device=dev)
    with torch.no_grad():
        video = diffusion.sample(x_cond=x_cond, task_embed=task_embed, batch_size=B)
    assert tuple(video.shape) == (B, CHANNELS, *TARGET_SIZE), video.shape
    frames = video.reshape(B, f, 3, *TARGET_SIZE)
    print(f"DDIM sample ok   : video={tuple(video.shape)} -> {f} frames {tuple(frames.shape[2:])}")

    print("PASS: AVDC ROCm env OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
