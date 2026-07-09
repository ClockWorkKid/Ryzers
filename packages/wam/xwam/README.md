### X-WAM

This package runs [X-WAM](https://github.com/sharinka0715/X-WAM) — *Unified 4D World Action
Modeling from Video Priors with Asynchronous Denoising* — a **Wan2.2-TI2V-5B** unified 4D
world-action model (UMT5-XXL text encoder + Wan2.2 VAE + video/depth/action DiT) on AMD Ryzen
AI Max+ 395 (Strix Halo, `gfx1151`) under ROCm 7.2.2. Direct PyTorch port: upstream code runs
on the base image's ROCm torch; the CUDA torch / flash-attn pins are dropped (attention falls
back to torch SDPA on gfx1151).

X-WAM jointly generates future **video + depth + robot states + actions**, and uses
**Asynchronous Noise Sampling (ANS)** — few action denoise steps for real-time control, full
steps for high-fidelity video — which is the headline efficiency lever this Ryzer targets.

It is a **slim policy/model layer that ships no simulator**, and a reference consumer of the
simulator packages' `Policy` seam. It composes on:

- the plain ROCm base &rarr; non-sim demos (smoke / latency / open-loop / videogen);
- the `simulation/robotwin` base &rarr; closed-loop + interactive RoboTwin 2.0;
- the `simulation/robocasa` base (new port) &rarr; closed-loop + interactive RoboCasa.

Weights (Wan2.2 base + X-WAM checkpoints) and datasets are fetched at runtime (rule 8); sim
assets come from the sim base.

> **Status: under development.** Phase 1 (image build + ROCm env sign-of-life) landing first;
> forward-pass / open-loop / closed-loop phases follow (see `docs/PLAN.md`).

### Build

```sh
# Standalone (env sign-of-life + non-sim demos):
ryzers build xwam --name xwam
ryzers run --name xwam                     # test.py: ROCm torch + GPU + X-WAM import sign-of-life

# Chain on a simulator base for closed-loop / interactive rollouts:
ryzers build robotwin xwam --name xwam-robotwin
ryzers build robocasa xwam --name xwam-robocasa
```

Artifacts are written to `workspace/xwam/outputs`. For faster/gated HF downloads set `HF_TOKEN`.

```sh
ryzers run --name xwam /ryzers/scripts/download_checkpoints.sh robotwin   # Wan2.2 base + robotwin_sft
EXP=robotwin_sft ryzers run --name xwam /ryzers/demos/demo_smoke.sh       # full-model action smoke
```

### Useful knobs

- `EXP` = `robotwin_sft` | `robocasa_sft` | `pretrained` — which checkpoint to load.
- `DENOISE_STEPS` (video, default 50) / `ACTION_DENOISE_STEPS` (action, default 10) — async denoising.
- `CFG`, `RUN_DEPTH`, `FRAME_NUM`, `SEED`, `PROMPT`.
- `HF_TOKEN` for faster/gated downloads.

### References

- Upstream: https://github.com/sharinka0715/X-WAM (pinned in `docs/UPSTREAM_PIN.commit.txt`)
- Paper: https://arxiv.org/abs/2604.26694 · Project: https://sharinka0715.github.io/X-WAM/
- Model: https://huggingface.co/sharinka0715/X-WAM-checkpoints · Wan2.2 base: https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B
- Datasets: https://huggingface.co/datasets/sharinka0715/X-WAM-RoboCasa · https://huggingface.co/datasets/sharinka0715/X-WAM-RoboTwin

Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
