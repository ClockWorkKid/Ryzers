### Nano World Model (NanoWM)

This package runs [Nano World Model](https://github.com/simchowitzlabpublic/nano-world-model)
— a minimalist **diffusion-forcing video world model** (Latte / DFoT / DINO-WM lineage) —
on AMD Ryzen AI Max+ 395 (Strix Halo, `gfx1151`) under ROCm 7.2.2. Direct PyTorch port:
upstream code runs on the base image's ROCm torch; the conda/CUDA torch stack is replaced
and only CUDA-specific pins are stripped.

NanoWM predicts **future video frames autoregressively** from a short history of frames
(encoded by a latent codec) plus optional actions — it is a pure **world model** (video
prediction), *not* text-conditioned and *not* an action-policy head. It therefore lives in
the new **`wm/`** category (world model), distinct from `wam/` (world-action) and `vla/`.

> **Status:** scaffold. Draft `Dockerfile`/`config.yaml`/`test.py` are placeholders pending
> Phase-1 validation on Strix Halo (see `docs/nano-world-model/PLAN.md`). Do not assume they
> build yet — dependency pins (esp. `pytorch-lightning==1.9.5`) are validated on the device.

### Build (target usage)

```sh
ryzers build nano-world-model --name nanowm
ryzers run --name nanowm                 # test.py: ROCm torch + GPU + NanoWM import sign-of-life
```

Weights/datasets are fetched at runtime into the mounted HF cache (never re-hosted, rule 8):

```sh
ryzers run --name nanowm /ryzers/scripts/download_checkpoints.sh   # a released HF checkpoint + codec
ryzers run --name nanowm /ryzers/scripts/download_datasets.sh      # a few dataset context episodes
```

### Demos

| Demo | Base | What it does |
|---|---|---|
| `demos/demo_rollout.sh` | plain | Load a checkpoint, autoregressive rollout → gen/gt/comparison mp4. |

```sh
DOMAIN=dino_wm_pusht NUM_SAMPLES=4 ROLLOUT_LENGTH=16 \
  ryzers run --name nanowm /ryzers/demos/demo_rollout.sh
```

### Released checkpoints (HF `knightnemo`)

Point Maze (30k) · Wall (15k) · Rope (15k) · Granular (15k) · PushT (100k) · RT-1 (300k) ·
CSGO (NanoWM-L/2, 100k). See `docs/nano-world-model/SCOPING.md` for the mapping.

### References

- Upstream: https://github.com/simchowitzlabpublic/nano-world-model (pinned in `docs/UPSTREAM_PIN.commit.txt`)
- Paper: https://arxiv.org/abs/2605.23993
- Checkpoints: https://huggingface.co/collections/knightnemo/nano-world-model

Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
