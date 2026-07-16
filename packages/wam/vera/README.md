### VERA

This package runs [VERA](https://github.com/sizhe-li/VERA) — *Turning Video Models into
Generalist Robot Policies* (MIT, 2026) — a **two-stage, closed-loop video-to-action policy** on
AMD Ryzen AI Max+ 395 (Strix Halo, `gfx1151`) under ROCm 7.2.2. Direct PyTorch port: upstream
code runs on the base image's ROCm torch; the CUDA `torch==2.6.0` pin is dropped and flash-attn is
omitted (the WAN planner falls back to torch SDPA / AOTriton on gfx1151).

VERA leaves a video generative model **as-is** as an action-free world model that "dreams" the
future, and trains an embodiment-specific **inverse-dynamics model (IDM)** on the robot
**Jacobian** to translate the dream into low-level actions:

- **Video planner** (`vera.video_model`) — action-free diffusion model that generates future
  frames from the current observation (+ optional text). Two backbones: **WAN** (Wan2.1 DiT) for
  MimicGen / DROID / OMNI, and a tiny **DFoT** U-Net3D flow planner for PushT.
- **Jacobian IDM** (`vera.idm` + `vera.policy`) — data-efficient dream→actions translator on the
  **VGGT-1B** visual backbone; embodiment-specific, swappable without retraining the planner.

It is a **slim planner+IDM+sim image**: `.[idm,video]` covers the video-generation path, and
`.[eval]` adds the MuJoCo / robosuite / gym-pusht stacks (+ EGL/OSMesa GL libs) for headless
closed-loop control. Weights and the frozen upstream bases (Wan2.1 T2V/I2V, VGGT-1B) are fetched at
runtime into a mounted HF cache (rule 8).

> **Status.** Video generation (Gates 1–3) and **closed-loop control (Phase 4)** landed:
> ROCm import smoke, WAN-14B DROID generation, and closed-loop rollouts for **both** embodiments —
> PushT (100% success) and MimicGen 2-block stacking (offscreen EGL). See `docs/PORT_SUMMARY.md`
> for per-gate results, `docs/CLOSEDLOOP.md` for the closed-loop repro, and `docs/PLAN.md` for the
> roadmap.

### Build

```sh
ryzers build vera --name vera          # builds on the ROCm 7.2.2 base
ryzers run --name vera                 # test.py: ROCm torch + GPU + VERA import sign-of-life
ryzers run --name vera python /ryzers/test_eval.py   # + MuJoCo/robosuite import & EGL offscreen render
```

Artifacts are written to `workspace/vera/outputs`. For faster/gated HF downloads set `HF_TOKEN`.

```sh
# download checkpoints into the mounted model volume (rule 8):
ryzers run --name vera /ryzers/scripts/download_checkpoints.sh wave1   # MimicGen + PushT, ~15 GB
ryzers run --name vera /ryzers/scripts/download_checkpoints.sh droid   # + DROID 14B planner, ~31 GB
```

### Demos

Weights + the frozen bases are fetched on first run. The closed-loop demos serve a live MJPEG
viewer over HTTP — tunnel the vis port with `ssh -L <PORT>:localhost:<PORT> <host>` and open
`http://localhost:<PORT>` (dashboard: current obs | dream+tracks | dream | Jacobian field). Each
run also dumps a `*_vis.mp4` of the viewer buffer.

| Demo | Needs | What it does |
|---|---|---|
| `demos/demo_smoke.sh` | plain | Env + VERA import smoke (no weights). |
| `demos/demo_videogen.sh` | DROID ckpts | DROID WAN-14B language-conditioned video generation (no sim). |
| `demos/demo_img2vid.sh` | DROID ckpts | Image→video: one still primed across the context window, planner dreams the rollout. |
| `demos/demo_pusht.sh` | wave1 + PushT zarr | PushT closed-loop (DFoT planner + Jacobian IDM, no MuJoCo). |
| `demos/demo_mimicgen.sh` | wave1 + `stack_d0` | MimicGen 2-block stack closed-loop (robosuite/MuJoCo, offscreen EGL). |

```sh
# PushT closed-loop (server + client + recorder in one container):
ryzers run --name vera /ryzers/demos/demo_pusht.sh
# MimicGen closed-loop — short viewer clip (1 demo × 50 steps ≈ 15 min on gfx1151):
NUM_DEMOS=1 ROLLOUT_HORIZON=50 ryzers run --name vera /ryzers/demos/demo_mimicgen.sh
```

### Useful knobs

- `PORT` / `VIS_PORT` — policy websocket + MJPEG viewer ports (PushT 8820/8821, MimicGen 8800/8801).
- MimicGen: `NUM_DEMOS`, `ROLLOUT_HORIZON`, `RENDER_SIZE`, `SAMPLE_STEPS` (denoise steps, default 10).
- PushT: `ZARR_PATH`, `FRAME_INDICES`, `N_REPEATS`, `HORIZON`, `SEED`.
- Video-gen: `TEXT`, `NUM_STEPS`, `SEED`, `IMAGE`/`VIEWS` (img2vid).
- `HF_TOKEN` for faster/gated downloads.

### References

- Upstream: https://github.com/sizhe-li/VERA (pinned in `docs/UPSTREAM_PIN.commit.txt`)
- Paper: *Turning Video Models into Generalist Robot Policies* (MIT, 2026) · Project: https://vera.csail.mit.edu
- Model: https://huggingface.co/sizhe-lester-li/VERA · Bases: `Wan-AI/Wan2.1-T2V-1.3B`,
  `Wan-AI/Wan2.1-I2V-14B-480P`, `facebook/VGGT-1B`

Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
