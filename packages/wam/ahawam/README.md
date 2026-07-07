### AHA-WAM

This package runs [AHA-WAM](https://github.com/serene-sivy/AHA-WAM) — an **asynchronous**
Wan2.2-TI2V-5B world-action model (umt5-xxl text encoder + Wan VAE + video-DiT world planner
+ action-DiT executor) — on AMD Ryzen AI Max+ 395 (Strix Halo, `gfx1151`) under ROCm 7.2.2.
Direct PyTorch port: upstream code runs on the base image's ROCm torch; only the CUDA torch
pins are stripped. AHA-WAM is the direct successor of FastWAM; this package mirrors
`packages/wam/fastwam`.

AHA-WAM splits inference into two phases that can run at different rates:

- `phase="video"` &rarr; Observation-Guided Video-Context prefill (the **slow planner**);
- `phase="action"` &rarr; one Action-DiT chunk (`action_chunk_size` control steps) that reuses
  the prefilled video KV cache (the **fast executor**).

It is a **slim policy/model layer that ships no simulator**, and the reference consumer of
the simulator packages' `Policy` seam. It composes on:

- the plain ROCm base &rarr; non-sim demos (smoke / latency / open-loop);
- the `simulation/robotwin` base &rarr; closed-loop RoboTwin 2.0.

The AHA-WAM install is pinned to the base image's torch + numpy, so the same policy layer
composes on a plain base (numpy 2.x) or the `simulation/robotwin` base (numpy 1.26.4).
AHA-WAM ships **RoboTwin 2.0 weights only** (no LIBERO checkpoint). Weights/datasets are
fetched by the scripts below; the ~12 GB Wan2.2 base is fetched automatically on the first
model run (DiffSynth, default source ModelScope).

### Build

```sh
# Standalone (non-sim demos + model sign-of-life):
ryzers build ahawam --name ahawam
ryzers run --name ahawam                  # test.py: ROCm torch + GPU + deps sign-of-life

# Chain on the RoboTwin 2.0 simulator base for closed-loop rollouts:
ryzers build robotwin ahawam --name ahawam-robotwin
```

Artifacts are written to `workspace/ahawam/outputs`. For faster/gated HF downloads set
`HF_TOKEN`.

```sh
ryzers run --name ahawam /ryzers/scripts/download_checkpoints.sh all   # base + Flash ckpts
ryzers run --name ahawam /ryzers/scripts/download_datasets.sh          # open-loop RoboTwin data
```

### Demos

| Demo | Base | What it does |
|---|---|---|
| `demos/demo_smoke.sh` | plain | Build the real model, load ckpt, one two-phase `infer_action`; cold/steady latency + VRAM. |
| `demos/demo_latency.sh` | plain | Two-phase latency: slow video prefill vs fast action chunk, executor control Hz + SDPA backends. |
| `demos/demo_openloop.sh` | plain | Replay GT RoboTwin episodes, overlay predicted vs GT action chunks + MAE. |
| `demos/demo_async_rt.sh` | plain | Async real-time serving: upstream `deploy/` TCP server (`--async-mode`) + async dummy client. |
| `demos/demo_closedloop_robotwin.sh` | `robotwin` | Closed-loop RoboTwin 2.0 rollouts (SAPIEN Vulkan RT) + success rate + videos. |

```sh
ryzers run --name ahawam /ryzers/demos/demo_smoke.sh
ryzers run --name ahawam /ryzers/demos/demo_async_rt.sh                 # WHICH=flash (default)
TASKS="click_bell lift_pot" NUM_EPISODES=10 \
  ryzers run --name ahawam-robotwin /ryzers/demos/demo_closedloop_robotwin.sh
```

The RoboTwin closed-loop runs RoboTwin's own `script/eval_policy.py` against the upstream
`experiments/robotwin/ahawam_policy` plugin (`EVALUATION.robotwin_root=/opt/RoboTwin`),
scheduling the two phases via `EVALUATION.chunks_per_video_prefill`. The async real-time
demo reuses the upstream `deploy/` stack unchanged (server + async dummy client).

### Results (Strix Halo, Radeon 8060S, gfx1151, ROCm 7.2.2)

- **Model smoke** — real 13.74 B-param model loads and runs both phases end-to-end;
  action chunk `(16, 14)`, steady-state ~1.6 s at 10 denoise steps, peak ~30.5 GB.
- **Open-loop replay** — predicted action chunks track ground truth: mean normalized MAE
  **0.0094**, raw-unit MAE **0.0060** (per-dim overlays confirm tight tracking on the
  large-motion arm joints).
- **Closed-loop RoboTwin 2.0** — `click_bell` **2/2 = 100%** with rendered rollout videos.
- **Async real-time** — Flash (`num_inference_steps=1`) async serving: 30/30 action
  requests, 0 errors, 648 image frames pushed concurrently on the decoupled channel.
  Two-phase latency: video prefill **407 ms** (amortized in background), action chunk
  **405 ms** &rarr; **~39.5 control Hz** executor throughput.

### Useful knobs

- Non-sim: `WHICH=flash|robotwin` (async/latency), `NUM_STEPS`, `SEED`, `NUM_EPISODES`.
- Async RT: `PORT`, `INSTRUCTION`, `NUM_ACTION_REQUESTS`, `ACTION_RATE`, `IMAGE_FPS`, `PREFILL_WAIT`.
- Closed-loop RoboTwin: `TASKS`, `TASK_CONFIG`, `NUM_EPISODES`, `CHUNKS_PER_VIDEO_PREFILL`, `NUM_INFERENCE_STEPS`.
- `CKPT`, `DATASET_STATS` to point at specific weights; `HF_TOKEN` for faster/gated downloads.

### References

- Upstream: https://github.com/serene-sivy/AHA-WAM (pinned in `docs/UPSTREAM_PIN.commit.txt`)
- Checkpoints: https://huggingface.co/SereneC/AHA-WAM-RoboTwin2.0
- Dataset: https://huggingface.co/datasets/yuanty/robotwin2.0-fastwam

Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
