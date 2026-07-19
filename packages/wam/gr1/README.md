### GR-1

This package runs [GR-1](https://github.com/bytedance/GR-1) — *Unleashing Large-Scale Video
Generative Pre-training for Visual Robot Manipulation* (ByteDance, ICLR 2024) — a **GPT-style
vision-language-action policy** on AMD Ryzen AI Max+ 395 (Strix Halo, `gfx1151`) under ROCm 7.2.2.
Direct PyTorch port: the upstream code runs on the base image's ROCm torch; upstream's CUDA
`torch==1.12.1` pin (`install.sh`) is dropped and only minimal ROCm/API source patches are applied
(rule 2.1 — rely on upstream, patch only what the platform needs).

GR-1 is a single autoregressive transformer that consumes a short observation history + a language
goal and predicts the next arm/gripper action *and* future frames (the video-pre-training objective
carried into fine-tuning). The forward path is all plain matmul+softmax attention — nothing
CUDA-specific in the math:

- **MAE ViT-B/16** image encoder (`models/vision_transformer.py`, in-repo) — static + gripper cameras.
- **OpenAI CLIP ViT-B/32** text encoder (`clip.load`) for the language goal.
- **Perceiver resampler** (`flamingo_pytorch`) compressing 196 patch tokens → 9 latents.
- **GPT-2 trajectory backbone** (`models/trajectory_gpt2.py`, vendored) over the token sequence.
- **Action head** (arm 6-DoF + gripper) and **MAE-style future-frame decoder** heads.

It lives in the **`wam/`** category (world-action models), alongside `vera` and `cosmos3`.

**Single self-contained image (model + sim, rule 0.4).** GR-1's only benchmark simulator is
**CALVIN** (there is no other sim in the upstream repo — the paper's other results are real-robot
hardware). The CALVIN PyBullet stack (`mees/calvin`: `calvin_env` + `calvin_agent` + `tacto`) is
baked in as its own Dockerfile layers on top of the model path and renders **headless via EGL on the
AMD iGPU** (Mesa/gfx1151). Sim + eval deps install under the same base torch/numpy constraint so the
ROCm torch stack is never clobbered; CALVIN's `numba` (unused), `pyhash` (avoided by the closed-loop
runner) and its `pytorch-lightning`/torch-1.13 training pins are skipped, and `hydra-core` is bumped
`1.1.1 → 1.3.2` (1.1.1 crashes on Python 3.12). Weights and the CALVIN dataset are fetched at runtime
into mounted volumes (never re-hosted, rule 8).

> **Status.** Full port landed on Strix Halo (`gfx1151`, ROCm): weight-free ROCm smoke, open-loop
> prediction on real CALVIN episodes, **closed-loop CALVIN control (8/8 debug validation tasks)**, and
> a Phase-4 inference-optimization pass (**fp16 + ROCm-flash SDPA = 6.8× faster, closed-loop 8/8,
> action quality unchanged**; +`torch.compile` = 8.7×). See `OPTIMIZATIONS.md` for the full
> speed/quality study and `docs/UPSTREAM_PIN.commit.txt` for the pinned upstream commits.

### Build

```sh
ryzers build gr1 --name gr1     # GR-1 model path + CALVIN sim baked into one image
ryzers run --name gr1           # test.py: ROCm torch + GPU + per-submodule & full GR-1 forward (no weights)
```

The smoke `test.py` needs no weights or sim: it proves a working ROCm torch on the iGPU, imports the
model-path deps, and runs the MAE ViT / Perceiver resampler / GPT-2 backbone and the full GR-1 policy
forward on-device (rule 2), exiting non-zero on any failure.

### Weights & dataset

Fetched at runtime into the mounted volumes (never re-hosted, rule 8):

```sh
# GR-1 policy (ABCD-D) + MAE ViT-B/16 backbone -> /models (CLIP ViT-B/32 auto-downloads on first use):
ryzers run --name gr1 /ryzers/scripts/download_checkpoints.sh          # WHICH=all (default); or mae|abcd|abc
# CALVIN *debug* subset (~1.3 GB) for open-loop replay + closed-loop debug eval -> /data
# (the full task_ABCD_D is intentionally NOT downloaded, rules 3/4):
ryzers run --name gr1 /ryzers/scripts/download_calvin_debug.sh
```

### Demos

Artifacts are written to `workspace/gr1/outputs`. All knobs are host env vars
(`VAR=... ryzers run --name gr1 /ryzers/demos/<demo>.sh`).

| Demo | Needs | What it does |
|---|---|---|
| `demos/demo_openloop.sh` | weights + debug dataset | **Phase 2** — loads real MAE + GR-1 weights, replays a real CALVIN episode window offline (no sim), dumps a GT-vs-pred action overlay + a two-column future-frame comparison (rule 2.a). |
| `demos/demo_calvin.sh` | weights + debug dataset | **Phase 3** — closed-loop CALVIN control in the headless PyBullet sim; rolls out the debug validation tasks and writes rollout GIFs + a success rate to `outputs/calvin`. |
| `demos/demo_bench.sh` | weights + debug dataset | **Phase 4** — times the closed-loop step path and reports latency + arm MAE for the env-selected optimization config → `outputs/bench`. |

```sh
# open-loop prediction (one annotated window):
WINDOW_IDX=0 MAX_STEPS=32 ryzers run --name gr1 /ryzers/demos/demo_openloop.sh
# closed-loop CALVIN (all 8 debug tasks, fast fp16+flash default):
ryzers run --name gr1 /ryzers/demos/demo_calvin.sh
# benchmark the fp32 reference vs the fast path:
GR1_AMP=off GR1_SDPA=0 GR1_TAG=fp32 ryzers run --name gr1 /ryzers/demos/demo_bench.sh
GR1_AMP=fp16 GR1_SDPA=1 GR1_TAG=fp16sdpa ryzers run --name gr1 /ryzers/demos/demo_bench.sh
```

### Useful knobs

- **Eval:** `NUM_TASKS` (0 = all debug tasks), `EP_LEN` (max closed-loop steps/task), `SPLIT`
  (`training`|`validation`), `SEED`; `WINDOW_IDX`/`MAX_STEPS` for open-loop.
- **Paths:** `POLICY_CKPT`, `MAE_CKPT`, `DATASET_DIR` (default `/data/calvin_debug_dataset`),
  `CALVIN_ROOT` (default the baked `/repos/calvin`).
- **Optimization** (`demos/gr1_optim.py`, see `OPTIMIZATIONS.md`) — image defaults to the fast path:
  `GR1_AMP` (`off`|`bf16`|`fp16`, default `fp16`), `GR1_SDPA` (`0`|`1`, default `1`),
  `GR1_FLASH` (pin the ROCm-native AOTriton FLASH backend, default `1`), `GR1_COMPILE` (`0`|`1`).

### Strix Halo optimization

Phase 4 speeds up per-step inference on the Radeon 8060S iGPU **without regressing action quality** —
every config is scored on latency *and* arm MAE by one harness (`demos/demo_bench.py`) and the winner
must hold the closed-loop 8/8. Opt-in, env-toggled layer over the *unmodified* upstream graph
(`demos/gr1_optim.py`). Headline (per-step, torch 2.10.0+rocm7.2.2):

| config | ms/step | fps | speedup | arm MAE | closed-loop |
|---|---|---|---|---|---|
| fp32 baseline | 389.7 | 2.57 | 1.0× | 0.0421 | 8/8 |
| **fp16 + SDPA (flash)** — default | **57.5** | **17.4** | **6.8×** | 0.0421 | **8/8** |
| fp16 + SDPA + compile | 45.0 | 22.2 | 8.7× | 0.0426 | 8/8 |

ROCm-native flash attention on gfx1151 (RDNA3.5) is delivered **through torch SDPA's AOTriton FLASH
backend** — the standalone `flash_attn`/composable-kernel library is CDNA-only and has no gfx1151
kernels. `GR1_FLASH=1` pins that backend (`FLASH → mem-efficient → math`) so the flash kernel runs
deterministically and never silently regresses to the ~17× slower math path. Full study, backend
probe, and findings in `OPTIMIZATIONS.md`.

### ROCm / portability patches

Minimal, idempotent, string-replacement based so they survive small upstream drift
(`patches/rocm_port.py`, `patches/calvin_port.py`; applied after the pip layers):

- `models/gr1.py` — device-agnostic query masks (`torch.zeros(...).cuda()` → `device=rgb.device`).
- `evaluation/calvin_evaluation.py` — `map_location='cpu'` on the policy `torch.load`.
- `models/trajectory_gpt2.py` — transformers 4.36.2 API drift (`tokenizer_class`→`processor_class`,
  `n_ctx`→`n_positions`) in the vendored GPT-2.
- `calvin_env/.../camera.py` — cast camera RGB to `uint8` (pybullet 3.2.7 + numpy 2.x returns int64).

### References

- Upstream: https://github.com/bytedance/GR-1 · CALVIN sim: https://github.com/mees/calvin
  (both pinned in `docs/UPSTREAM_PIN.commit.txt`)
- Paper: *Unleashing Large-Scale Video Generative Pre-training for Visual Robot Manipulation*
  (ICLR 2024) · https://arxiv.org/abs/2312.13139
- Weights: GR-1 ABCD-D / ABC-D snapshots (ByteDance), MAE ViT-B/16 (`facebookresearch/mae`),
  CLIP ViT-B/32 (OpenAI) — all fetched at runtime.

Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
