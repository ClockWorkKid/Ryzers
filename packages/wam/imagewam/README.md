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

All demos are `ryzers run` wrappers over `scripts/` using in-container paths (`/models`,
`/repos`, `/outputs`) and env overrides; they check for their prerequisites (weights / sim
base / mounted dataset) and print a hint if missing. Outputs land in `workspace/*/outputs`.

| Demo | Base | What it does |
|---|---|---|
| `demos/demo_smoke.sh` | plain | Load checkpoint, one `infer_action`; cold/steady latency + VRAM (`VISUALIZE_DREAM=1` also renders the dreamed frame). |
| `demos/demo_openloop_libero.sh` | plain + LIBERO dataset | **P5** open-loop on real episodes (dataset replay, no sim): action MAE vs GT, AE PSNR, dreamed stills (`ol_metrics.json`, `ol_action_overlay.png`, `ol_dream_*.png`). |
| `demos/demo_dreamvideo_openloop.sh` | plain + dataset | Continuous open-loop dream **video**: every frame is `[obs \| GT future \| dream]` (`DATASET=libero\|robotwin`; augmentation disabled for stable GT/obs columns). |
| `demos/demo_latency.sh` | plain | **P8** per-module latency/computation profile (action vs dream path, fixed-vs-per-step split, param distribution, measured text-cache win) -> `p8_latency.json` + `.png`. Synthetic shapes, no dataset/sim. |
| `demos/demo_closedloop_libero.sh` | `libero` | **P6** closed-loop LIBERO rollouts (MuJoCo, upstream evaluator) + `success_summary.json`. |
| `demos/demo_closedloop_robotwin.sh` | `robotwin` | **P6b** closed-loop RoboTwin 2.0 rollouts (SAPIEN, upstream `eval_policy`). |
| `demos/demo_dreamvideo_closedloop_libero.sh` | `libero` | Continuous **dream-vs-sim** video: actual sim frame every step (left) + re-dreamed future at each replan (right). |
| `demos/demo_dreamvideo_closedloop_robotwin.sh` | `robotwin` | RoboTwin analogue of the dream-vs-sim video (3-cam compact layout). |
| `demos/demo_interactive_libero{,_rt}.sh` | `libero` | **P7** interactive LIBERO server — synchronous and real-time. |
| `demos/demo_interactive_robotwin{,_rt}.sh` | `robotwin` | **P7** interactive RoboTwin server — synchronous and real-time. |

### Latency & optimization (P8)

Profiled on **Radeon 8060S (Strix Halo `gfx1151`)**, bf16, `num_inference_steps=20`,
`action_horizon=16`, input 224×448 (`scripts/profile_modules.py`; full writeup + chart on the
laptop under `artifacts/latency/`). The deployment cost is the **action path**
(`infer_action_flux2`); the **dream path** (`infer_video_flux2`) is image-world-model /
visualization only and never runs during control.

| Path | Total / call | Dominant cost |
|---|---:|---|
| Action (control) | **1071 ms** | prefill KV 357 ms (33%) + action MoT loop 453 ms (42%, 22.7 ms/step) + Qwen3 195 ms (18%) |
| Dream (viz only) | **10757 ms** | video DiT loop 10344 ms (**96%**, 517 ms/step) — **~10× the action path, ~23× per step** |

- **Fixed cost ≈ 605 ms (57%)** of the action path; the diffusion-step knob has limited
  leverage (20→5 steps saves only ~33%) — the key ImageWAM finding: control never pays for the
  3.88B video expert generation loop (84% of params), only its conditioning prefill.
- **Text-embedding caching across replans: measured −18% (−193 ms → 878 ms).** The instruction
  is fixed per episode; `infer_action_flux2` already accepts `context`/`context_mask`, so encode
  Qwen3 once and reuse — a drop-in, exactly-lossless win.
- **`torch.compile[default]` of the per-step MoT action forward: ~1.25× on the diffusion loop**
  (max|Δ| ~1e-3 vs eager, bf16). The loop is compute/bandwidth-bound, not launch-bound —
  HIP-graph capture gave 0.99× and `reduce-overhead` regressed, so inductor default is used.
  Explored but not adopted: ParaDiGMS parallel-in-time sampling (near-lossless but only ~1.09×
  here — the large video KV prefix makes per-step attention scale with batch). ROCm knob sweep
  confirmed `TORCH_BLAS_PREFER_HIPBLASLT=0` + bf16 is optimal (hipBLASLt=1 regresses ~32%,
  fp16 NaNs). Remaining levers: shrink the `Sv≈905` video prefix, weight-only fp8/int8 experts.

**Baked into the default route.** Both proven wins ship ON by default for every demo via
`scripts/imagewam_opt.py` (patches the `ImageWAM`/`MoT` classes) — wired through
`scripts/opt_launch.py` for the upstream closed-loop evaluators and interactive servers, and
inline in the open-loop/dream scripts. No upstream files are modified. They fall back to eager on
any error and are env-gated: `IMAGEWAM_OPT=0` (all off), `IMAGEWAM_TEXT_CACHE=0`,
`IMAGEWAM_ACTION_COMPILE=0`. Note: `torch.compile` pays a one-time warm-up (~1–2 min) on the first
action step of a process; set `IMAGEWAM_ACTION_COMPILE=0` if that stall is undesirable (e.g. a
short real-time demo). `demos/demo_latency.sh` is intentionally left unwrapped so its A/B
(naive vs text-cached) stays a clean measurement.

### Status

Port complete on branch `wam-imagewam` (off `benchmark`). Milestones: P0 scoping ✓ ·
P1 scaffold ✓ · P2 import smoke ✓ · P3 full-model ROCm smoke ✓ · P4 module validation ✓ ·
P5 open-loop + dream ✓ · P6 closed-loop (LIBERO + RoboTwin 2.0) ✓ · P7 interactive
(sync + real-time) ✓ · P8 latency/opt analysis ✓.

### References

- Upstream: https://github.com/yuyangalin/ImageWAM (pinned in `docs/UPSTREAM_PIN.commit.txt`)
- Backbone: https://github.com/black-forest-labs/flux2 (pinned)
- Models: https://huggingface.co/collections/yuyangalin/imagewam
- Datasets (shared with FastWAM): https://huggingface.co/datasets/yuanty/LIBERO-fastwam · https://huggingface.co/datasets/yuanty/robotwin2.0-fastwam

Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
