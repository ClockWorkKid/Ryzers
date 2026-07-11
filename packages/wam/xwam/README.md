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

> **Status.** Gates 1–4 landed: ROCm env sign-of-life, full-model forward, open-loop eval,
> and closed-loop + interactive on **both** sim bases (RoboTwin 2.0 and RoboCasa). See
> `docs/PORT_SUMMARY.md` for per-gate results and `docs/PLAN.md` for the roadmap.

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

### Demos

Weights + sim assets are fetched on first run. The interactive demos serve a browser page over
HTTP/MJPEG — tunnel the port with `ssh -L <PORT>:localhost:<PORT> <host>` and open
`http://localhost:<PORT>`. Each drives X-WAM through the sim base's model-agnostic `Policy`
seam (`POLICY_FACTORY`); the closed-loop runners write per-episode videos + a `_result.json`
success rate.

| Demo | Base | What it does |
|---|---|---|
| `demos/demo_smoke.sh` | plain | Full-model action smoke (no sim). |
| `demos/demo_openloop.sh` | plain | Open-loop GT-vs-pred action replay + MAE plot. |
| `demos/demo_closedloop_robotwin.sh` | robotwin | Closed-loop RoboTwin 2.0 rollouts (dual-arm EE via IK). |
| `demos/demo_closedloop_robocasa.sh` | robocasa | Closed-loop RoboCasa kitchen rollouts (single-arm OSC_POSE). |
| `demos/demo_interactive_robotwin.sh` | robotwin | Interactive RoboTwin (chunk-replay), `PORT=8082`. |
| `demos/demo_interactive_robotwin_rt.sh` | robotwin | Real-time RoboTwin (async planner; arms HOLD while thinking), `PORT=8083`. |
| `demos/demo_interactive_robocasa.sh` | robocasa | Interactive RoboCasa (chunk-replay), `PORT=8082`. |
| `demos/demo_interactive_robocasa_rt.sh` | robocasa | Real-time RoboCasa (async planner + HOLD), `PORT=8083`. |

```sh
# e.g. interactive RoboCasa on the chained image:
TASK=TurnOnSinkFaucet ryzers run --name xwam-robocasa /ryzers/demos/demo_interactive_robocasa.sh
# closed-loop RoboCasa, 10 randomized episodes:
TASK=TurnOnSinkFaucet NUM_EVALS=10 ryzers run --name xwam-robocasa /ryzers/demos/demo_closedloop_robocasa.sh
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
