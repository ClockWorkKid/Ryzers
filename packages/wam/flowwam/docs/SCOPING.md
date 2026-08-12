# FlowWAM - Upstream Scoping (Ryzers port, Strix Halo / gfx1151, ROCm 7.2.2)

> **Correction (post-P5):** this doc originally scoped only the world-model repo
> `FlowWAM_WorldArena` and left "flow → action for closed loop" as open Q1. That question is now
> resolved: the **full `FlowWAM` repo** (https://github.com/YixiangChen515/FlowWAM) ships the **IDM
> action expert** + a RoboTwin flow-action policy server/client - that is the genuine closed loop
> (now staged; see `PORT_SUMMARY.md` §6 and `PLAN.md` P6). The autoregressive world-model rollout is
> *not* closed-loop. The package therefore uses **two upstream repos**: `FlowWAM_WorldArena`
> (open-loop world model) and `FlowWAM` (closed-loop action policy).

Open-loop upstream: https://github.com/YixiangChen515/FlowWAM_WorldArena
Closed-loop upstream: https://github.com/YixiangChen515/FlowWAM
Paper: *FlowWAM: Optical Flow as a Unified Action Representation for World Action Models* (arXiv 2607.13017)

## 1. Thesis / what makes FlowWAM unique
FlowWAM is a **video-diffusion World-Action Model** whose central idea is to use **dense optical
flow as a unified action representation**. Instead of regressing joint/EE actions with a separate
action head (FastWAM) or predicting a single edited goal image (ImageWAM), FlowWAM:
1. generates a **future RGB video** conditioned on a reference frame + language instruction, then
2. extracts **RAFT optical flow** between frames, encodes it with a **reversible flow codec** - flow
   is the action carrier,
3. renders a **robot-only frame** from RoboTwin2.0 embodiment URDFs via **SAPIEN** to disentangle
   robot motion from scene, and
4. (closed-loop) feeds generated flow to an **IDM action expert** that outputs 14-D robot actions.

Per rule 2.b, open-loop viz shows the reference/GT frame(s) on the **left** and the generated/dreamed
video on the **right**; the **flow field** is the FlowWAM-specific first-class visualization target.

## 2. Model stack (all Wan-lineage - big reuse from FastWAM)
| Component | Source | Role |
|---|---|---|
| **Wan2.2-TI2V-5B** | `Wan-AI/Wan2.2-TI2V-5B` | dual-stream world-model DiT backbone |
| **UMT5-XXL text enc** | `Wan-AI/Wan2.1-T2V-1.3B` (`models_t5_umt5-xxl-enc-bf16.pth` + `google/` tok) | language conditioning |
| **Wan VAE** | bundled w/ Wan2.2-TI2V-5B | video latent enc/dec |
| **FlowWAM world-model ckpt** | `YixiangChen/FlowWAM` `flowwam_worldarena_stage1.safetensors` | open-loop world model |
| **FlowWAM action ckpt** | `YixiangChen/FlowWAM` `flowwam_robotwin.safetensors` + action-norm stats | closed-loop policy (IDM) |
| **SeedVR2-3B refiner** | `ByteDance-Seed/SeedVR2-3B` | stage-2 refinement - **deferred** (apex CUDA-only) |
| **RAFT** | torchvision | optical-flow estimator |
| **RoboTwin2.0 embodiments** | `TianxingChen/RoboTwin2.0` `embodiments.zip` | SAPIEN robot URDFs |

The library is a **trimmed DiffSynth-Studio** (`diffsynth/`), the same Wan-video framework FastWAM
builds on.

## 3. Datasets / simulators
- **WorldArena / RoboTwin2.0** is the open-loop benchmark; `packages/simulation/robotwin` (SAPIEN +
  Vulkan headless on gfx1151) is the base - FlowWAM's SAPIEN robot renderer + embodiments reuse it.
- Open-loop test split: `WorldArena_Robotwin2.0/test_dataset` (1000 episodes). Closed-loop:
  standard RoboTwin 2.0 `aloha-agilex` tasks (50-task suite).

## 4. ROCm porting notes (reuse the FastWAM playbook wholesale)
- torch/torchvision CUDA pins → **strip** (`strip_cuda_torch.py`) and hold base ROCm torch via
  `PIP_CONSTRAINT` (numpy<2 pinned to base).
- **flash_attn** (CUDA-only) → DiffSynth attention falls back to `torch.nn.functional.
  scaled_dot_product_attention` on ROCm (rule 2.1).
- **apex** + **SeedVR2 refiner** → CUDA-only, deferred (stage-1/world-model gen is the deliverable).
- **nvidia-cublas-cu12** → drop; **pynvml** → stub/patch for ROCm.
- **sapien** → Vulkan renderer, already working on gfx1151 via the RoboTwin base; reuse it.

## 5. Compliance (rules 8, 9)
- All weights downloaded by scripts/Dockerfile at build/run time - **never re-hosted** in the PR.
- DiffSynth/SeedVR are vendored **in upstream** (Apache-2.0); we clone upstream at pinned commits in
  the Dockerfile, not vendor them ourselves.
- **No AMD-internal access details in any committed file.**

## 6. Package placement
- `packages/wam/flowwam/` (mirrors `packages/wam/fastwam/`). Slim model layer, ships NO simulator;
  composes on `simulation/robotwin`. Work branch `wam-flowwam` off `origin/benchmark`.
