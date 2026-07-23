### FlowWAM

This package runs **FlowWAM** — a Wan2.2-TI2V-5B **dual-stream world-action model** that uses
**dense optical flow as a unified action representation** — on AMD Ryzen AI Max+ 395 (Strix Halo,
`gfx1151`) under ROCm 7.2.2. FlowWAM jointly generates a future RGB video and a flow field
(dual-stream Wan DiT), extracts RAFT optical flow through a reversible flow codec, and renders
robot-only frames from RoboTwin 2.0 embodiments via SAPIEN.

**Two evaluation modes (both merged into this package):**

| Mode | What it is | Upstream repo | Checkpoint | Driver |
|------|-----------|---------------|-----------|--------|
| **Open-loop (world model)** | Predict future video given first frame + action-derived flow; score vs GT video. **No controller.** | [`FlowWAM_WorldArena`](https://github.com/YixiangChen515/FlowWAM_WorldArena) | `flowwam_worldarena_stage1.safetensors` | `scripts/open_loop_eval.py`; long-horizon rollout stress-test in `scripts/wm_autoregressive_eval.py` |
| **Closed-loop (action policy)** | **Genuine control loop**: model → IDM action expert → RoboTwin `env.step`, replanning each chunk; score task success. | [`FlowWAM`](https://github.com/YixiangChen515/FlowWAM) | `flowwam_robotwin.safetensors` + `..._action_norm_stats.npz` | ws server + `robotwin_policy` via RoboTwin's `eval_policy.py` (`demos/demo_closedloop_robotwin.sh`) |

> Note: `wm_autoregressive_eval.py` chains the world model on its own frames — this is an
> autoregressive *world-model* rollout, **not** closed-loop control (no action feedback). Only the
> closed-loop mode above drives the robot.

Direct PyTorch port: upstream (a trimmed DiffSynth) runs on the base image's ROCm torch. Only
the base-owned CUDA/torch/numpy/sapien pins are stripped; **no** flash-attn/apex/cuBLAS are
installed — DiffSynth's `flash_attention()` falls back to torch SDPA on ROCm, and apex is only
needed by the (deferred) SeedVR2 refiner.

It is a **slim model layer that ships no simulator**. Because it requires SAPIEN robot rendering
+ RoboTwin embodiments, it composes on the **`simulation/robotwin`** base:

```sh
# Chain on the RoboTwin 2.0 sim base (SAPIEN/Vulkan headless on gfx1151):
ryzers build robotwin flowwam --name flowwam-robotwin
ryzers run   --name flowwam-robotwin           # test.py: ROCm torch + Wan dual-stream env smoke
```

### Weights

Everything is fetched from the original HF repos on demand (never re-hosted). The Wan2.2-TI2V-5B
base (~12 GB) + FlowWAM stage-1 checkpoint + RoboTwin embodiments:

```sh
ryzers run --name flowwam-robotwin /ryzers/scripts/download_checkpoints.sh all
# or piecewise: base | stage1 (world model) | robotwin (action policy) | embodiments
```

For gated repos / the RoboTwin embodiments dataset, set `HF_TOKEN`.

### Demos

```sh
ryzers run --name flowwam-robotwin /ryzers/demos/demo_smoke.sh                 # env sign-of-life
ryzers run --name flowwam-robotwin /ryzers/demos/demo_closedloop_robotwin.sh   # genuine closed loop
# open-loop world-model eval: scripts/open_loop_eval.py (+ wm_autoregressive_eval.py long-horizon)
```

Artifacts are written under `workspace/flowwam/outputs`. The SeedVR2 refiner is intentionally
out of this image (its apex dependency is CUDA-only); the deliverable is stage-1 video gen.

### References

- Open-loop (world model): https://github.com/YixiangChen515/FlowWAM_WorldArena (pinned in `docs/UPSTREAM_PIN.commit.txt`)
- Closed-loop (action policy): https://github.com/YixiangChen515/FlowWAM
- Paper: *FlowWAM: Optical Flow as a Unified Action Representation for World Action Models* (arXiv 2607.13017)

Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
