# WAN 2.2 TI2V-5B Docker Setup

This Ryzer package runs the WAN 2.2 TI2V-5B video pipeline on AMD ROCm systems,
supporting both **text-prompted (t2v)** and **image-prompted (i2v)** generation.
It uses the upstream WAN 2.2 source with a ROCm `flash_attn` shim and a
feathered-overlap tiled VAE decode for Strix Halo (gfx1151 / Radeon 8060S).

Weights are not baked into the image. They are downloaded once into the
persistent `/models` volume and reused for prompt-driven generation.

## Fast Route (Production Default)

The package ships a lossless "fast route" enabled by default (override any lever
via the matching host env var before `ryzers run`):

- `WAN22_VAE_BF16=1` — bf16 VAE decode (~3x faster than fp32, no visible loss).
- `TILE_H=12 TILE_W=12 STRIDE_H=8 STRIDE_W=8` — 12x12 latent tiles with a 4-cell
  (64px) overlap that is **feather-blended** so tile seams are invisible. The
  feather window also down-weights the artifact-prone tile borders where the VAE
  decoder lacked cross-tile latent context.
- `VAE_TILE_BATCH=8` — decode 8 tiles per batched VAE call.
- `WAN22_TEXT_KV_CACHE=1` — bit-exact text / cross-attention KV cache.
- `WAN22_VAE_OFFLOAD=1` — for i2v, free/offload the VAE after the pre-loop
  `encode()` so its full-resolution encoder activations do not fragment VRAM and
  starve the DiT self-attention. Without this, long i2v clips (121f) OOM even
  though the identical t2v DiT forward fits.

See `FAILURE_MODES.md` for optimization levers that were measured and rejected
(torch.compile, token merging, whole-video batching, sparse conv, temporal-delta
decode).

## Build And Smoke Test

Build the image:

```sh
ryzers build wan22
```

Run the default smoke test:

```sh
ryzers run
```

The smoke test verifies ROCm PyTorch, SDPA attention, the `flash_attn` shim, WAN
TI2V config import, CLI entry points, and basic `.mp4` writing. It does not
download weights or run generation.

## Download Weights

Download `Wan-AI/Wan2.2-TI2V-5B` once:

```sh
ryzers run /ryzers/download_wan22.sh
```

If Hugging Face authentication is required, set `HF_TOKEN` before running the
download command.

## Generate Video

Generate a 121-frame, 1280x704 video:

```sh
PROMPT="A dreamy nighttime sky filled with glowing stars over a quiet village. One giant star suddenly flickers, sneezes loudly, and falls from the sky wearing fuzzy pajamas and bunny slippers. Villagers below stare in confusion while a dog howls at it. Soft magical lighting, fantasy animation style, cozy but absurd." \
SIZE=1280x704 \
FRAME_NUM=121 \
ryzers run /ryzers/demo_wan22.sh
```

The generated `.mp4` is written under `/outputs`, which maps to
`$PWD/workspace/wan22/outputs` on the host.

Use `SAVE_FILE` to choose the output path:

```sh
PROMPT="A realistic brown fox jumps over a sleeping dog in a grassy field." \
SIZE=1280x704 \
FRAME_NUM=121 \
SAVE_FILE=/outputs/foxdog.mp4 \
ryzers run /ryzers/demo_wan22.sh
```

If setting variables on separate lines, export them first:

```sh
export PROMPT="A cinematic landscape shot of waves crashing at sunrise."
export SIZE=1280x704
export FRAME_NUM=121
ryzers run /ryzers/demo_wan22.sh
```

## Image-Assisted Generation (i2v)

Provide `IMAGE` to seed generation from a conditioning image. The first frame is
anchored to the image and the prompt drives the motion. Mode is auto-detected
when `IMAGE` is set (force with `MODE=i2v`).

```sh
PROMPT="Two anthropomorphic cats in boxing gear trade rapid combinations on a spotlighted stage, dynamic motion, cinematic lighting." \
IMAGE=/outputs/i2v_condition.png \
SIZE=1280x704 \
FRAME_NUM=121 \
SAMPLE_STEPS=50 \
SAVE_FILE=/outputs/i2v_clip.mp4 \
ryzers run /ryzers/demo_wan22.sh
```

Measured on Strix Halo (Radeon 8060S, ROCm 7.14) for a full 5 s (121f / 50-step)
1280x704 i2v clip on the fast route: **~62 min** total (DiT ~55 min, feathered
bf16 tiled VAE decode ~6.6 min). t2v of the same size/steps is comparable.

Supported generation variables:

- `PROMPT` or `PROMPT_FILE`
- `IMAGE`: conditioning image path for i2v (enables image-prompted mode)
- `MODE`: `auto` (default), `t2v`, or `i2v`
- `SIZE`: `1280x704` or `704x1280`
- `FRAME_NUM`: explicit WAN frame count, must be `4n+1`
- `DURATION_SECONDS`: duration rounded up to a valid `4n+1` frame count
- `SAMPLE_STEPS`: default `50`
- `SEED`: default `20260525`
- `SAVE_FILE`: default auto-generated under `/outputs`
- `BENCHMARK_JSON`: optional path to write per-stage timing + optimization flags

Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.

