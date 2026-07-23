# FlowWAM — Ryzers Port Summary (adoption · benchmark · tuning)

**Target:** AMD Ryzen AI Max+ 395 "Strix Halo" (`gfx1151`), ROCm 7.2.2, `torch 2.10.0+rocm7.2.2`
(hip 7.2.53211), Radeon 8060S iGPU, single GPU. Composed on the `simulation/robotwin` base
(SAPIEN Vulkan headless). This doc is the presentation/reporting entry point; deeper scope and the
milestone log are in `SCOPING.md` and `PLAN.md`.

## 1. What FlowWAM is
A Wan2.2-TI2V-5B **dual-stream World-Action Model** that uses **dense optical flow as a unified
action representation**. One shared dual-stream Wan DiT both (a) *conditions on* flow to model the
world (predict future video) and (b) *generates* flow that an **IDM action expert** turns into robot
actions. It ships **two evaluation modes**, both merged into this single package:

| Mode | Loop | Upstream repo | Checkpoint | Driver |
|------|------|---------------|-----------|--------|
| **Open-loop (world model)** | predict future video from first frame + action-derived flow; score vs GT video. **No controller.** | `FlowWAM_WorldArena` | `flowwam_worldarena_stage1.safetensors` | `scripts/open_loop_eval.py` |
| **Closed-loop (action policy)** | **genuine control:** model → IDM action expert → RoboTwin `env.step`, replan each chunk; score task success. | `FlowWAM` | `flowwam_robotwin.safetensors` + action-norm stats | ws server + `robotwin_policy` via RoboTwin `eval_policy.py` (`demos/demo_closedloop_robotwin.sh`) |

## 2. Port / adoption (ROCm playbook) — no source patches needed
- Reuse the FastWAM Wan/DiffSynth ROCm playbook wholesale: strip base-owned CUDA/torch/numpy/sapien
  pins from `requirements.txt`, hold the base ROCm torch via `PIP_CONSTRAINT`; DiffSynth
  `flash_attention()` falls back to **torch SDPA** on ROCm; **apex + the SeedVR2 refiner are skipped**
  (CUDA-only). Verified: flash3=flash2=sage=False, bf16 Wan-attention op runs on-device.
- Two upstream repos share a `diffsynth` package: the open-loop world-model repo is `pip install -e`
  into the image; the closed-loop server uses the full FlowWAM repo's `diffsynth` via runtime
  `PYTHONPATH` so the validated open-loop install is untouched (rules 0.4 / 2.1).
- Image: `flowwam-robotwin` layered on `simulation/robotwin` (weights fetched at run time, never
  re-hosted — rules 8/9).

## 3. Per-module validation (rule 2) — all PASS on gfx1151
- **Model load:** 11.39B params (text_encoder 5.68B + DiT 5.00B + VAE 0.705B + flow_stream 1.19M),
  built in ~13 s; DiT dim=3072, ffn=14336, 30 layers, 24 heads, in/out=48, patch [1,2,2].
- **UMT5-XXL text enc** → context `(1,512,4096)` bf16, finite.
- **Wan2.2 VAE** enc+dec roundtrip on a real image → **MSE 0.006** (`artifacts/flowwam/p4_vae_roundtrip.png`).
- **Dual-stream DiT** forward → rgb_pred + flow_pred `(1,48,2,16,16)`, finite.
- **RAFT + reversible flow codec** → RAFT flow correct; codec encode→decode **MAE 0.067 px** (near-lossless).
- **SAPIEN robot-only renderer** → 460 ms/frame on ROCm/Vulkan (dual-arm aloha-agilex qpos replay).

## 4. Open-loop world-model eval (WorldArena RoboTwin 2.0) — DONE
- 384×288, 15 denoise steps, single 33-frame chunk anchored to the **real** first frame → faithful
  reproduction of scene layout (objects, table, arm). Two-column `GT | dream` (rule 2.b) + flow strip
  + 10-episode 3-panel `[REAL OBS | FLOW (action) | DREAM]` in `artifacts/flowwam/{openloop,threepanel}`.
- **Timing:** one-time VAE warmup ~525 s (first-call ROCm kernel compile), then ~43–68 s/episode.
  VAE **decode is the dominant ROCm cost**.
- **Constraint found:** dual-stream needs rgb+flow latents with identical spatial dims; VAE latent =
  W,H/16 diverges when a latent dim is odd → **input W,H must be divisible by 32** (the runner snaps).

## 5. Long-horizon autoregressive world-model rollout (stress test — *not* closed-loop)
`scripts/wm_autoregressive_eval.py` chains the world model on its own frames (last dreamed frame
anchors the next chunk). This is an **open-loop-family** stress test with **no controller / no action
feedback** — it is *not* closed-loop control. Process-isolated run (PSNR, dream vs GT):

| Episode | chunks | PSNR first→last | mean |
|---|---|---|---|
| 51 | 2 | 45 → 20 | 22.6 |
| 69 | 3 | 48 → 6 | 13.0 |
| 590 | 3 | 24 → 16 | 12.3 |
| 199 | 3 | 22 → 20 | 14.2 |
| 7 | 1 | 8.7 → 6.7 | 8.1 |

**Finding:** the first chunk (≤33 frames, real-anchored) is faithful (20–48 dB); quality degrades
stepwise at each autoregressive hand-off. Consistency is a function of **re-anchoring to real
frames** — which is exactly what the genuine closed loop does (§6), so closed-loop is expected to
avoid this drift. Divergence curves per episode in `artifacts/flowwam_wm_autoregressive/`.

## 6. Closed-loop action policy (RoboTwin) — STAGED (authored, pending GPU validation)
Genuine control loop using the **upstream FlowWAM flow-action pipeline**, adapted for ROCm on our
robotwin base (RoboTwin's stock `script/eval_policy.py`; **no sim source edits** — policy symlinked
as `policy/flowwam`):

```
RoboTwin env --obs(head/left/right RGB + qpos + instruction)--> flow_action_server
  DiT flow-conditioned video latents -> IDM action expert -> 14-D action chunk --ws:8000-->
  robotwin_policy client --> env.step ... replan every EXECUTE_WINDOW(25) from a FRESH real obs
```

Server↔client are co-located in one venv on the single GPU (upstream splits them across two conda
envs). **Remaining to validate (needs GPU):** rebuild image with the closed-loop layer → confirm the
flow-action server loads on gfx1151 → run RoboTwin task(s) → record success rates. Not yet run.

## 7. Tuning / optimization roadmap (bf16 baseline; VAE decode is the prime target)
Flow-conditioned early exit on ~static regions; reversible-flow-codec caching across rollouts;
robot-only render memoization (scene-independent given joint state); UMT5-text + reference-VAE-latent
memoization across rollouts; low-res flow path (RAFT + codec at 320×240, upsample only for viz);
RAFT-vs-lighter flow estimator profiling. Diffusion-step count is out of scope.

## 8. Reproduce
```sh
ryzers build robotwin flowwam --name flowwam-robotwin
ryzers run --name flowwam-robotwin /ryzers/scripts/download_checkpoints.sh all   # base|stage1|robotwin|embodiments
ryzers run --name flowwam-robotwin /ryzers/demos/demo_smoke.sh                    # env sign-of-life
ryzers run --name flowwam-robotwin /ryzers/demos/demo_closedloop_robotwin.sh      # closed loop (needs GPU)
# open-loop world-model eval: scripts/open_loop_eval.py ; long-horizon: scripts/wm_autoregressive_eval.py
```

Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved. SPDX-License-Identifier: MIT
