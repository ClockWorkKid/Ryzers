# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Per-module validation for FlowWAM on AMD Strix Halo (gfx1151), per workspace rule 2:
exercise each submodule *separately* with real/partial data and the fully-loaded weights before
running the end-to-end pipeline. Uses the upstream call surface (rule 2.1), not reimplementations:

  1. UMT5-XXL text encoder : pipe.prompter.encode_prompt(prompt) -> context.
  2. Wan2.2 VAE            : encode + decode roundtrip on a REAL image; report recon MSE.
  3. Wan2.2 dual-stream DiT: one model_fn_wan_video_dual_stream() forward on small latents.
  4. RAFT + FlowCodec      : optical flow on two real frames, then the *reversible* flow codec
                             encode->decode roundtrip; report flow round-trip error (the FlowWAM
                             action-representation core).

Each test is independent; a failure is reported but the others still run. Exits non-zero if any
critical module fails. Env: FLOWWAM_REPO, FLOWWAM_MODEL_DIR, FLOWWAM_CKPT (see model_smoke.py).
Writes a couple of small artifacts (VAE recon + flow codec viz) under OUT_DIR for inspection.
"""
import os
import sys
import time
import traceback

import numpy as np
import torch
from PIL import Image

FLOWWAM_REPO = os.environ.get("FLOWWAM_REPO", "/repos/flowwam")
MODEL_DIR = os.environ.get("FLOWWAM_MODEL_DIR", "/models/flowwam")
CKPT = os.environ.get(
    "FLOWWAM_CKPT", os.path.join(MODEL_DIR, "stage_1", "flowwam_worldarena_stage1.safetensors"))
OUT_DIR = os.environ.get("OUT_DIR", "/outputs")
PROMPT = os.environ.get("PROMPT", "the robot arm picks up the block and places it in the box")
# A real image that ships in the upstream tree (SeedVR asset) -> "real data" for VAE + flow.
REAL_IMG = os.environ.get(
    "REAL_IMG", os.path.join(FLOWWAM_REPO, "inference/refiner/SeedVR/assets/teaser.png"))

results = {}


def _real_image(size=(256, 256)):
    if os.path.exists(REAL_IMG):
        img = Image.open(REAL_IMG).convert("RGB").resize(size)
        print(f"  using real image: {REAL_IMG} -> {img.size}")
        return img
    print(f"  real image not found ({REAL_IMG}); using synthetic gradient")
    x = np.linspace(0, 255, size[0], dtype=np.uint8)
    arr = np.stack([np.tile(x, (size[1], 1)), np.tile(x[::-1], (size[1], 1)),
                    np.tile(x, (size[1], 1)).T], axis=-1)
    return Image.fromarray(arr)


def test_text_encoder(pipe):
    pipe.load_models_to_device(["text_encoder"])
    t0 = time.time()
    ctx = pipe.prompter.encode_prompt(PROMPT, positive=True, device=pipe.device)
    dt = time.time() - t0
    t = ctx[0] if isinstance(ctx, (list, tuple)) else ctx
    assert torch.is_tensor(t) and torch.isfinite(t.float()).all(), "non-finite text embedding"
    print(f"  context: {tuple(t.shape)} {t.dtype} on {t.device}  ({dt:.2f}s)")
    return ctx


def test_vae(pipe):
    img = _real_image((256, 256))
    pipe.load_models_to_device(["vae"])
    ref = pipe.preprocess_image(img).transpose(0, 1)          # (C,1,H,W) upstream layout
    t0 = time.time()
    z = pipe.vae.encode([ref], device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
    enc_dt = time.time() - t0
    assert torch.isfinite(z.float()).all(), "non-finite VAE latent"
    t1 = time.time()
    vid = pipe.vae.decode(z, device=pipe.device, tiled=False)
    frames = pipe.vae_output_to_video(vid)
    dec_dt = time.time() - t1
    recon = frames[0].resize(img.size)
    a = np.asarray(img, np.float32) / 255.0
    b = np.asarray(recon, np.float32) / 255.0
    mse = float(np.mean((a - b) ** 2))
    print(f"  latent {tuple(z.shape)} {z.dtype}  encode {enc_dt:.2f}s decode {dec_dt:.2f}s  "
          f"recon MSE={mse:.5f}")
    try:
        os.makedirs(OUT_DIR, exist_ok=True)
        w = img.size[0]
        combo = Image.new("RGB", (w * 2, img.size[1]))
        combo.paste(img, (0, 0)); combo.paste(recon, (w, 0))     # GT | recon (rule 2.b spirit)
        combo.save(os.path.join(OUT_DIR, "p4_vae_roundtrip.png"))
    except Exception as e:  # noqa: BLE001
        print(f"  (viz skipped: {e})")
    assert mse < 0.05, f"VAE recon MSE too high ({mse:.4f}); loading/dtype likely broken"
    return z


def test_dit(pipe, flow_stream, context):
    from diffsynth.pipelines.wan_video_dual_stream import model_fn_wan_video_dual_stream
    vae_z_dim = getattr(pipe.vae, "z_dim", 16)
    up = pipe.vae.upsampling_factor
    height = width = 256
    num_frames = 5
    latent_length = (num_frames - 1) // 4 + 1
    h, w = height // up, width // up
    shape_z = (1, vae_z_dim, latent_length, h, w)
    print(f"  latent shape {shape_z} (z_dim={vae_z_dim}, up={up})")
    rgb_latents = pipe.generate_noise(shape_z, seed=0, rand_device="cpu").to(
        dtype=pipe.torch_dtype, device=pipe.device)
    flow_z = pipe.generate_noise(shape_z, seed=1, rand_device="cpu").to(
        dtype=pipe.torch_dtype, device=pipe.device)
    pipe.scheduler.set_timesteps(50, shift=5.0)
    ts = pipe.scheduler.timesteps[0].unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
    pipe.load_models_to_device(pipe.in_iteration_models)
    t0 = time.time()
    rgb_pred, flow_pred = model_fn_wan_video_dual_stream(
        dit=pipe.dit, flow_stream=flow_stream, latents=rgb_latents, flow_latents=flow_z,
        timestep=ts, context=context, fuse_vae_embedding_in_latents=True,
        use_gradient_checkpointing=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.time() - t0
    assert rgb_pred.shape == rgb_latents.shape, f"rgb_pred shape {rgb_pred.shape}"
    assert torch.isfinite(rgb_pred.float()).all(), "non-finite rgb_pred"
    print(f"  dual-stream forward: rgb_pred {tuple(rgb_pred.shape)} "
          f"flow_pred {None if flow_pred is None else tuple(flow_pred.shape)}  ({dt:.2f}s)")


def test_flow_codec():
    sys.path.insert(0, os.path.join(FLOWWAM_REPO, "inference"))
    from video_flow_codec_pipeline import RAFTFlowExtractor
    from reversible_flow_codec import FlowCodec
    img = _real_image((320, 240))
    f1 = np.asarray(img, np.uint8)
    f2 = np.roll(f1, shift=6, axis=1)                 # induce a known ~6px horizontal flow
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    raft = RAFTFlowExtractor(device=dev)
    t0 = time.time()
    flow = raft(f1, f2)                                # (H,W,2)
    dt = time.time() - t0
    assert flow.shape[:2] == f1.shape[:2] and flow.shape[2] == 2, f"flow shape {flow.shape}"
    assert np.isfinite(flow).all(), "non-finite RAFT flow"
    mean_dx = float(np.mean(flow[..., 0]))
    print(f"  RAFT flow {flow.shape} mean_dx={mean_dx:+.2f}px (expect ~ -6)  ({dt:.2f}s)")
    # Reversible codec: encode flow -> RGB -> decode -> flow'. Check round-trip error.
    # Upstream encode() returns (rgb, max_magnitude); reuse the codec's own auto max_magnitude
    # (percentile-based) so this mirrors real FlowWAM usage.
    codec = FlowCodec()
    rgb, max_mag = codec.encode(flow)                 # max_magnitude=None -> auto (99.5 pct)
    flow2 = codec.decode(rgb, max_magnitude=max_mag)
    clipped = np.clip(flow, -max_mag, max_mag)        # codec clips magnitude to max_mag
    rt_err = float(np.mean(np.abs(clipped - flow2)))
    print(f"  auto max_magnitude={max_mag:.2f}px")
    print(f"  FlowCodec roundtrip: rgb {rgb.shape} {rgb.dtype}  |flow-decode| MAE={rt_err:.4f}px")
    try:
        os.makedirs(OUT_DIR, exist_ok=True)
        Image.fromarray(rgb.astype(np.uint8)).save(os.path.join(OUT_DIR, "p4_flow_encoded.png"))
    except Exception as e:  # noqa: BLE001
        print(f"  (viz skipped: {e})")
    assert rt_err < 1.0, f"flow codec not reversible enough (MAE {rt_err:.3f}px)"


def main() -> int:
    print(f"torch {torch.__version__} hip={torch.version.hip}")
    if not (torch.version.hip and torch.cuda.is_available()):
        print("FAIL: need a ROCm device", file=sys.stderr); return 1
    print(f"device: {torch.cuda.get_device_name(0)}")
    for p in (FLOWWAM_REPO, os.path.join(FLOWWAM_REPO, "inference")):
        if p not in sys.path:
            sys.path.insert(0, p)
    from world_model_inference import build_pipeline

    device = torch.device("cuda")
    t0 = time.time()
    pipe, flow_stream = build_pipeline(device, CKPT, local_model_path=MODEL_DIR)
    print(f"pipeline built: {time.time()-t0:.1f}s\n")

    context = None
    tests = [
        ("1. UMT5-XXL text encoder", lambda: results.__setitem__("context", test_text_encoder(pipe))),
        ("2. Wan2.2 VAE roundtrip", lambda: test_vae(pipe)),
        ("3. Wan2.2 dual-stream DiT forward", lambda: test_dit(pipe, flow_stream, results.get("context"))),
        ("4. RAFT + reversible FlowCodec", test_flow_codec),
    ]
    ok = {}
    for name, fn in tests:
        print(f"===== {name} =====")
        try:
            fn(); ok[name] = True; print(f"  -> PASS\n")
        except Exception as e:  # noqa: BLE001
            ok[name] = False
            print(f"  -> FAIL: {type(e).__name__}: {e}"); traceback.print_exc(); print()

    print("===== SUMMARY =====")
    for name, _ in tests:
        print(f"  [{'PASS' if ok.get(name) else 'FAIL'}] {name}")
    if all(ok.values()):
        print("PASS: all FlowWAM modules validated on ROCm"); return 0
    print("FAIL: one or more modules failed", file=sys.stderr); return 1


if __name__ == "__main__":
    raise SystemExit(main())
