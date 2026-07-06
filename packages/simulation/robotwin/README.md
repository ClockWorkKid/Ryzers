### simulation/robotwin

Model-agnostic [RoboTwin 2.0](https://github.com/RoboTwin-Platform/RoboTwin) simulator base
image for AMD Ryzen AI Max+ 395 (Strix Halo, `gfx1151`) under ROCm 7.2.2.

This package ships **only the simulator**: the SAPIEN (offscreen Vulkan ray-tracing) + mplib
motion-planning closed-loop stack plus a small harness (`sim_robotwin`) that exposes a
model-agnostic `Policy` seam and generic interactive / sanity runners. It contains **no**
policy or model code. Any VLA/WAM (FastWAM, ...) is layered on top as a separate ryzers
image and selected at runtime — see *Plug in your own policy* below.

RoboTwin is **de-vendored**: cloned from upstream at a pinned commit into `/opt/RoboTwin` at
build time and patched (`patches/robotwin_rocm.patch`). The patch bundles the ROCm changes
(curobo &rarr; mplib, OIDN denoiser off, optional open3d, expert seed pre-check off) with
model-agnostic eval-harness adaptations to `script/eval_policy.py` and `envs/_base_task.py`
(configurable output dir / episode count / skip-obs, and the 4-view eval-frame compositor
the harness streams). Sim assets are third-party and fetched at run time by
`scripts/setup_robotwin.sh`, never baked into the image.

### Build

```sh
ryzers build robotwin --name sim-robotwin
ryzers run --name sim-robotwin                       # test.py: ROCm + SAPIEN/mplib import sign-of-life
ryzers run --name sim-robotwin /ryzers/demos/demo_sim_sanity.sh   # RandomPolicy 4-view rollout -> MP4
```

The first run downloads RoboTwin assets (embodiments/objects/configs, ~5 GB) into the
`/sim_models` mount; subsequent runs reuse them. Sanity videos land under
`workspace/simulation-robotwin/outputs` (mounted at `/sim_outputs`).

### Plug in your own policy (model entrypoint)

The **interactive** showcase is driven through one small interface, so any model connects
the same way. A policy implements `sim_robotwin.Policy`:

```python
class Policy:
    replan_steps = 8      # env steps executed per predicted chunk
    def reset(self, instruction): ...
    def predict_action_chunk(self, obs, instruction) -> np.ndarray:  # [T, 14] qpos rows
```

`obs` is the raw RoboTwin observation dict (`obs["observation"][cam]["rgb"]` for
head/left/right cameras, `obs["joint_action"]["vector"]` for the current 14-D qpos). The
returned chunk is execute-ready joint-space rows
`[left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]` (aloha-agilex), run via
`TASK_ENV.take_action(row, action_type="qpos")`.

Connect a model in three steps (no edits to this package):

1. Build your ryzer **FROM** the sim base, installing your model under the base's
   torch+numpy pins (RoboTwin/mplib need `numpy 1.26.4`; FastWAM's
   `scripts/strip_cuda_torch.py` is a working `PIP_CONSTRAINT` reference):
   ```sh
   ryzers build robotwin <yourmodel> --name <yourmodel>-robotwin
   ```
2. Ship an adapter implementing `Policy` — copy [`examples/template_policy.py`](examples/template_policy.py)
   or see the shipped FastWAM adapter (`wam/fastwam/adapters/fastwam_robotwin_policy.py`).
3. Select it at runtime:
   ```sh
   POLICY_FACTORY=<module>:build_policy ryzers run --name <yourmodel>-robotwin \
     /ryzers/demos/demo_interactive.sh          # http://localhost:8082
   ```

Unset `POLICY_FACTORY` &rarr; the built-in `RandomPolicy`.

> The parity-critical **closed-loop** eval keeps RoboTwin's own model-agnostic runner
> (`script/eval_policy.py` + `policy/<name>/deploy_policy.py`) unchanged; a chained policy
> drops its `policy/<name>` plugin into `/opt/RoboTwin/policy` (see FastWAM's
> `demo_closedloop_robotwin.sh`). The `Policy` seam above is for the interactive/sanity
> showcase.

### Demos

| Demo | What it does |
|---|---|
| `demos/demo_sim_sanity.sh` | Headless `RandomPolicy` 4-view rollout &rarr; `/sim_outputs` MP4 (no model). |
| `demos/demo_interactive.sh` | Command-driven browser demo over HTTP/MJPEG (`RandomPolicy` default). |
| `demos/demo_interactive_rt.sh` | Real-time browser demo (async planner; arm HOLDs while thinking). |

### Useful knobs

- `POLICY_FACTORY=<module>:<fn>` selects the policy (unset &rarr; `RandomPolicy`).
- `TASK`, `TASK_CONFIG` (`demo_clean` / `demo_randomized`), `SEED` pick + reproduce a scene.
- `PORT` changes the browser port; `MAX_STEPS` caps the episode length.

### Layout

```
packages/simulation/robotwin
  Dockerfile              # ROCm base -> SAPIEN/mplib + de-vendored RoboTwin + sim_robotwin
  config.yaml             # ryzers manifest (gpu, Vulkan, /sim_outputs + /sim_models mounts)
  test.py                 # import sign-of-life (no render, no model)
  patches/robotwin_rocm.patch   # ROCm compat + eval-harness adapt, applied to fresh clone @ pin
  scripts/setup_robotwin.sh     # fetch assets/configs -> /opt/RoboTwin, planner -> mplib
  examples/template_policy.py   # COPY-ME model-entrypoint adapter (docs, not baked)
  lib/sim_robotwin/       # harness (installed to /opt/sim, on PYTHONPATH)
    policy.py             #   Policy ABC + load_policy() (POLICY_FACTORY seam)
    random_policy.py      #   built-in RandomPolicy (default, no weights)
    taskenv.py            #   RoboTwin task-env wrapper (config resolution + step/render)
    rollout.py            #   model-agnostic chunk-replay episode loop
    render.py             #   MJPEG/JPEG, 4-view rescale, banner, even-dim MP4
    sanity.py             #   headless RandomPolicy rollout -> MP4
    interactive_server.py #   command-driven HTTP/MJPEG demo (chunk-replay)
    interactive_server_rt.py  # real-time HTTP/MJPEG demo (async planner + HOLD)
  demos/                  # demo_sim_sanity.sh, demo_interactive.sh, demo_interactive_rt.sh
```

Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
