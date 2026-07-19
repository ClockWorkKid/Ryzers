### Micro-World

This package runs [AMD-AGI/Micro-World](https://github.com/AMD-AGI/Micro-World) — a **Wan2.1 /
VideoX-Fun-derived, action-controlled interactive world model** (T2W 1.3B + I2W 14B, Minecraft-style
first-person control) — on AMD Ryzen AI Max+ 395 (Strix Halo, `gfx1151`) under ROCm 7.2.2. Direct
PyTorch port: upstream code runs on the base image's ROCm torch; flash-attn is omitted (the DiT
attention and the CLIP image encoder fall back to torch SDPA on gfx1151, see `patches/rocm_port.py`).
Same Wan2.1 lineage as VERA, so the `benchmark` optimization recipe transfers.

Four inference paths, each an unmodified upstream `examples/wan2.1/predict_*` reproduced via the
env-driven `demos/mw_generate.py` (no in-place upstream edits):

- **t2v** — text → video (Wan2.1-T2V-1.3B baseline, no action control).
- **t2w** — text → world, **action-controlled** (`WanActionControlNetModel` on the 1.3B base).
- **i2v** — image → video (Wan2.1-I2V-14B-480P inpaint, CLIP-primed).
- **i2w** — image → world, **action-controlled** (`WanActionAdaLNModel` + LoRA on the 14B base).

Action string format: `[[end_frame, "w s a d shift ctrl _ mouse_y mouse_x"], ..., "space_frames"]`
(the trailing string marks free/idle frames); it is auto-clamped to the effective clip length.

> **Status.** Gates 1–3 (import smoke, per-component load, all 4 examples headless) and Phase 4
> (optimization) landed. See `docs/PORT_SUMMARY.md` for per-gate results and
> `RUNTIME_OPTIMIZATION.md` for the measured optimization A/B.

### Build

```sh
ryzers build --name micro-world micro-world   # RyzerManager auto-prepends ryzer_env (ROCm torch base)
ryzers run --name micro-world                 # test.py: ROCm torch + GPU + full Micro-World import smoke
```

Artifacts are written to `workspace/micro-world/outputs`. For faster/gated HF downloads set `HF_TOKEN`.

### Weights (fetched at runtime into the mounted model volume, rule 8)

```sh
# WHICH selects the subset (base weights are large; present dirs are verified + skipped):
WHICH=t2v ryzers run --name micro-world /ryzers/scripts/download_checkpoints.sh   # Wan2.1-T2V-1.3B
WHICH=t2w ryzers run --name micro-world /ryzers/scripts/download_checkpoints.sh   # + amd/Micro-World-T2W (default)
WHICH=i2v ryzers run --name micro-world /ryzers/scripts/download_checkpoints.sh   # Wan2.1-I2V-14B-480P
WHICH=i2w ryzers run --name micro-world /ryzers/scripts/download_checkpoints.sh   # + amd/Micro-World-I2W
# mw-t2w / mw-i2w fetch only the AMD adapters (reuse an already-present base); all = everything.
```

### Demos

| Demo | Needs | Output (rule 2.b) |
|---|---|---|
| `demos/demo_t2v.sh` | `WHICH=t2v` | Text→video, single-column MP4. |
| `demos/demo_t2w_action.sh` | `WHICH=t2w` | Text→world action-controlled, single-column MP4 (HUD overlay). |
| `demos/demo_i2v.sh` | `WHICH=i2v` | Image→video, two-column: reference (left) \| generated (right). |
| `demos/demo_i2w_action.sh` | `WHICH=i2w` | Image→world action-controlled, two-column reference \| generated. |

```sh
ryzers run --name micro-world /ryzers/demos/demo_t2v.sh
# 14B/18B image paths default to model_cpu_offload (full residency exceeds the ~47 GB GPU pool):
GPU_MEMORY_MODE=model_cpu_offload ryzers run --name micro-world /ryzers/demos/demo_i2w_action.sh
```

### Useful knobs (override from the host: `VAR=... ryzers run --name micro-world /ryzers/demos/demo_*.sh`)

- **Generation:** `PROMPT`, `NEGATIVE_PROMPT`, `ACTION_LIST` (json), `REF_IMAGE`, `SAMPLE_SIZE` (`H,W`),
  `VIDEO_LENGTH`, `NUM_STEPS`, `GUIDANCE_SCALE`, `SAMPLER` (`Flow_Unipc`/`Flow_DPM++`/`Flow`), `SHIFT`,
  `SEED`, `FPS`.
- **Memory:** `GPU_MEMORY_MODE` (`model_full_load` / `model_cpu_offload` / `sequential_cpu_offload`).
  The 1.3B t2v/t2w paths run `model_full_load` (peak ~18 GB). The 14B/18B i2v/i2w paths default to
  `model_cpu_offload` because full residency exceeds the box's **~47 GB GPU-addressable memory**
  (32 GB VRAM carveout + GTT; the 96 GB is total system RAM) — offload encodes then evicts T5/CLIP and
  keeps the DiT resident for the denoise loop.
- **Speed (see `RUNTIME_OPTIMIZATION.md`):** `TEACACHE` (default on), `TEACACHE_THRESHOLD` (per-mode
  0.10/0.20), `NUM_SKIP_START_STEPS`, `CFG_SKIP_RATIO`, `MW_DISABLE_CUDNN` (conv override, default 1).
- **Weights:** `TRANSFORMER_PATH`, `LORA_PATH`, `LORA_WEIGHT`, `BASE_MODEL`. `HF_TOKEN` for downloads.

### Optimization A/B (reproduce the numbers)

```sh
# component profile + conv/steps/teacache/kv A/B on the cheap t2v-1.3B proxy -> mw_opt_ab.json:
ryzers run --name micro-world bash -lc 'cd /repos/micro-world && python /ryzers/scripts/opt_ab_microworld.py'
```

### References

- Upstream: https://github.com/AMD-AGI/Micro-World (pinned in `docs/UPSTREAM_PIN.commit.txt`)
- Model: `amd/Micro-World-T2W`, `amd/Micro-World-I2W` · Bases: `Wan-AI/Wan2.1-T2V-1.3B`,
  `Wan-AI/Wan2.1-I2V-14B-480P`

Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
