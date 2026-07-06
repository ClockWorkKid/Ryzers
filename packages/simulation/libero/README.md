### simulation/libero

Model-agnostic [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) simulator base
image for AMD Ryzen AI Max+ 395 (Strix Halo, `gfx1151`) under ROCm 7.2.2.

This package ships **only the simulator**: the LIBERO / MuJoCo / robosuite closed-loop
stack (headless EGL) plus a small harness (`sim_libero`) that exposes a model-agnostic
`Policy` seam and generic closed-loop / interactive / sanity runners. It contains **no**
policy or model code. Any VLA/WAM (FastWAM, MolmoAct2, VLA-JEPA, ...) is layered on top as
a separate ryzers image and selected at runtime — see *Plug in your own policy* below.

### Build

```sh
ryzers build libero --name sim-libero
ryzers run --name sim-libero                       # test.py: ROCm + LIBERO import sign-of-life
ryzers run --name sim-libero /ryzers/demos/demo_sim_sanity.sh   # RandomPolicy rollout -> MP4
```

The built-in `RandomPolicy` needs no weights and proves the env renders + steps. Sanity
videos land under `workspace/simulation-libero/outputs` (mounted at `/sim_outputs`). LIBERO
bddl/init assets are baked into the image; nothing large otherwise.

### Plug in your own policy (model entrypoint)

The simulator is driven through one small interface, so any model connects the same way
the upstream LIBERO benchmark takes a pluggable policy. A policy implements
`sim_libero.Policy`:

```python
class Policy:
    replan_steps = 5      # env steps executed per predicted chunk
    num_steps_wait = 5    # no-op settle steps at episode start
    def reset(self, instruction): ...
    def predict_action_chunk(self, obs, instruction) -> np.ndarray:  # [T, 7] OSC_POSE delta + gripper
```

`obs` is the raw LIBERO/robosuite observation dict (`agentview_image`,
`robot0_eye_in_hand_image`, `robot0_eef_pos/quat`, `robot0_gripper_qpos`, ...). The harness
owns the env, rendering, MJPEG streaming and the episode loop; your policy only maps one
`(obs, instruction)` to an action chunk.

Connect a model in three steps (no edits to this package):

1. Build your ryzer **FROM** the sim base, installing your model under the base's
   torch+numpy pins (LIBERO needs `numpy 1.26.4`; FastWAM's `scripts/strip_cuda_torch.py`
   is a working `PIP_CONSTRAINT` reference):
   ```sh
   ryzers build libero <yourmodel> --name <yourmodel>-libero
   ```
2. Ship an adapter implementing `Policy` — copy [`examples/template_policy.py`](examples/template_policy.py)
   (annotated skeleton) or see [`examples/vjepa_libero_policy.py`](examples/vjepa_libero_policy.py)
   (VLA-JEPA-shaped stub) and the shipped FastWAM adapter (`wam/fastwam/adapters/fastwam_libero_policy.py`).
3. Select it at runtime:
   ```sh
   POLICY_FACTORY=<module>:build_policy ryzers run --name <yourmodel>-libero \
     /ryzers/demos/demo_interactive.sh          # http://localhost:8080
   ```

Unset `POLICY_FACTORY` &rarr; the built-in `RandomPolicy`.

### Demos

| Demo | What it does |
|---|---|
| `demos/demo_sim_sanity.sh` | Headless `RandomPolicy` rollout &rarr; `/sim_outputs` MP4 (no model). |
| `demos/demo_interactive.sh` | Command-driven browser demo over HTTP/MJPEG (`RandomPolicy` default). |
| `demos/demo_interactive_rt.sh` | Real-time browser demo (async planner; arm HOLDs while thinking). |

Chained policy packages add their own model-driven demos (e.g. FastWAM's
`demo_interactive_libero.sh`, `demo_closedloop_libero.sh`).

### Useful knobs

- `POLICY_FACTORY=<module>:<fn>` selects the policy (unset &rarr; `RandomPolicy`).
- `SUITE`, `TASK_ID`, `SEED` make runs reproducible.
- `PORT` changes the browser port; `VIEW_RES` / `VIDEO_RES` size the live/saved video.

### Layout

```
packages/simulation/libero
  Dockerfile              # ROCm base -> LIBERO stack + sim_libero harness
  config.yaml             # ryzers manifest (gpu, EGL env, /sim_outputs mount, POLICY_FACTORY)
  test.py                 # import sign-of-life (no render, no model)
  examples/               # COPY-ME model-entrypoint adapters (docs, not baked)
    template_policy.py     #   annotated Policy skeleton
    vjepa_libero_policy.py #   VLA-JEPA-shaped stub
  lib/sim_libero/         # harness (installed to /opt/sim, on PYTHONPATH)
    policy.py             #   Policy ABC + load_policy() (POLICY_FACTORY seam)
    random_policy.py      #   built-in RandomPolicy (default, no weights)
    libero_env.py         #   vendored LIBERO glue (env, image, dummy action, horizons)
    scene.py              #   (suite, task_id) scene wrapper
    rollout.py            #   model-agnostic chunk-replay episode loop
    render.py             #   MJPEG/JPEG, composed view, banner, even-dim MP4
    sanity.py             #   headless RandomPolicy rollout -> MP4
    interactive_server.py #   command-driven HTTP/MJPEG demo (chunk-replay)
    interactive_server_rt.py  # real-time HTTP/MJPEG demo (async planner + HOLD)
  demos/                  # demo_sim_sanity.sh, demo_interactive.sh, demo_interactive_rt.sh
```

Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
