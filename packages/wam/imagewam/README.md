### ImageWAM

This package runs [ImageWAM](https://github.com/yuyangalin/ImageWAM) — *"Do World Action
Models Really Need Video Generation, or Just Image Editing?"* — on AMD Ryzen AI Max+ 395
(Strix Halo, `gfx1151`) under ROCm 7.2.2. ImageWAM replaces video generation with a single
**image-editing** step: the model "dreams" one edited future frame and conditions its action
head on it. We port the **FLUX.2 [klein] ImageWAM** variant (Qwen3 text encoder + FLUX.2
autoencoder + image-editing DiT + ActionDiT) — the recommended/strongest variant and the
only one with released checkpoints. Direct PyTorch port: upstream code runs on the base
image's ROCm torch; only the CUDA (cu118) torch pins are stripped and `transformers` is
bumped to the FLUX.2-compatible `4.56.1`.

It is a **slim policy/model layer that ships no simulator**, built on FastWAM's framework, so
it reuses FastWAM's preprocessed LIBERO/RoboTwin datasets and the simulator packages' `Policy`
seam. It composes on:

- the plain ROCm base &rarr; non-sim demos (smoke / weights / open-loop / dream);
- the `simulation/libero` base &rarr; closed-loop + interactive LIBERO;
- the `simulation/robotwin` base &rarr; closed-loop RoboTwin 2.0.

### Build

```sh
# Standalone (non-sim demos + model sign-of-life):
ryzers build imagewam --name imagewam
ryzers run --name imagewam                 # test.py: ROCm torch + GPU + deps sign-of-life

# Chain on a simulator base for closed-loop / interactive rollouts:
ryzers build libero   imagewam --name imagewam-libero
ryzers build robotwin imagewam --name imagewam-robotwin
```

Artifacts are written to `workspace/*/outputs`. **FLUX.2 base + AE weights are gated**
(`black-forest-labs`) — set `HF_TOKEN` (with granted access) before fetching them.

```sh
HF_TOKEN=... ryzers run --name imagewam /ryzers/scripts/download_checkpoints.sh libero 4b
```

This fetches the public ImageWAM checkpoint (`yuyangalin/ImageWAM-FLUX.2-4B-LIBERO`:
`model.pt` + `dataset_stats.json` + `train_config.yaml`), the gated FLUX.2 klein-base-4B DiT
+ FLUX.2-dev AE, and (on first model run) the Qwen3-4B text encoder.

### Demos

| Demo | Base | What it does |
|---|---|---|
| `demos/demo_smoke.sh` | plain | Load checkpoint, one `infer_action`; cold/steady latency + VRAM (`VISUALIZE_DREAM=1` also renders the dreamed frame). |

*(open-loop / dream / latency / closed-loop / interactive demos land as milestones P5–P8 — see `../../../../docs` / laptop `PLAN.md`.)*

### Status

Port in progress on branch `wam-imagewam` (off `benchmark`). Milestones: P0 scoping ✓ ·
P1 scaffold ✓ · P2 import smoke · P3 weight-download smoke · P4 module validation ·
P5 open-loop + dream · P6 closed-loop · P7 interactive · P8 latency/opt analysis.

### References

- Upstream: https://github.com/yuyangalin/ImageWAM (pinned in `docs/UPSTREAM_PIN.commit.txt`)
- Backbone: https://github.com/black-forest-labs/flux2 (pinned)
- Models: https://huggingface.co/collections/yuyangalin/imagewam
- Datasets (shared with FastWAM): https://huggingface.co/datasets/yuanty/LIBERO-fastwam · https://huggingface.co/datasets/yuanty/robotwin2.0-fastwam

Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
