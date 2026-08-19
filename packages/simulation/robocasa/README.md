### simulation/robocasa

Model-agnostic [RoboCasa](https://github.com/robocasa/robocasa) (24 kitchen manipulation
tasks, RoboSuite v1.5 / MuJoCo backend) simulator base image for AMD Ryzen AI Max+ 395
(Strix Halo, `gfx1151`) under ROCm 7.2.2.

This package ships **only the simulator**: the RoboCasa / robosuite / MuJoCo closed-loop
stack (headless EGL) plus a small harness (`sim_robocasa`) that exposes a model-agnostic
`Policy` seam and generic closed-loop / interactive / sanity runners. It contains **no**
policy or model code. Any VLA/WAM (X-WAM, ...) is layered on top as a separate ryzers image
and selected at runtime — see *Plug in your own policy* below.

### Build

```sh
ryzers build robocasa --name sim-robocasa
ryzers run --name sim-robocasa                       # test.py: ROCm + RoboCasa import sign-of-life
ryzers run --name sim-robocasa /ryzers/demos/demo_sim_sanity.sh   # RandomPolicy rollout -> MP4
```

`robosuite` (master, RoboCasa v0.2 requirement) and `robocasa` are cloned at pinned commits
into the image (code only). The **~10GB kitchen assets are fetched at run time** by
`scripts/setup_robocasa.sh` into the mounted volume (`workspace/simulation-robocasa/assets`),
never baked into the image (first sanity/interactive run triggers the one-time download).
Sanity videos land under `workspace/simulation-robocasa/outputs` (mounted at `/sim_outputs`).

### Plug in your own policy (model entrypoint)

The simulator is driven through one small interface. A policy implements
`sim_robocasa.Policy`:

```python
class Policy:
    replan_steps = 32     # env steps executed per predicted chunk before replanning
    num_steps_wait = 0    # RoboCasa needs no settle steps
    def reset(self, instruction): ...
    def predict_action_chunk(self, obs, instruction) -> np.ndarray:  # [T, 7] OSC_POSE delta + gripper
```

`obs` is the harness observation dict: `obs["video"]` (`[V,H,W,3]` float32 in `[-1,1]`, the
3 cameras agentview_left/agentview_right/eye_in_hand), `obs["proprios"]` (`[16]` = eef
xyz + quat wxyz + gripper + 8 zeros), `obs["view"]` (stitched display frame), and
`obs["instruction"]`. RoboCasa executes the 7-D delta straight through robosuite's OSC_POSE
composite controller — no IK / motion-planning seam — so the same `Policy` drives both the
closed-loop rollout and the interactive demos. The harness owns the env, rendering, MJPEG
streaming and the episode loop; your policy only maps one `(obs, instruction)` to a chunk.

Connect a model in three steps (no edits to this package):

1. Build your ryzer **FROM** the sim base, installing your model under the base's
   torch+numpy pins (RoboCasa needs `numpy 1.26.4`):
   ```sh
   ryzers build robocasa <yourmodel> --name <yourmodel>-robocasa
   ```
2. Ship an adapter implementing `Policy` — copy [`examples/template_policy.py`](examples/template_policy.py)
   (annotated skeleton), or see the shipped X-WAM adapter
   (`wam/xwam/experiments/robocasa_xwam/xwam_policy/`).
3. Select it at runtime:
   ```sh
   POLICY_FACTORY=<module>:build_policy ryzers run --name <yourmodel>-robocasa \
     /ryzers/demos/demo_interactive.sh          # http://localhost:8082
   ```

Unset `POLICY_FACTORY` &rarr; the built-in `RandomPolicy`.

### Demos

| Demo | What it does |
|---|---|
| `demos/demo_sim_sanity.sh` | Headless `RandomPolicy` rollout &rarr; `/sim_outputs` MP4 (no model). |
| `demos/demo_interactive.sh` | Command-driven browser demo over HTTP/MJPEG (`RandomPolicy` default). |
| `demos/demo_interactive_rt.sh` | Real-time browser demo (async planner; arm HOLDs while thinking). |

Chained policy packages add their own model-driven demos (e.g. X-WAM's
`demo_closedloop_robocasa.sh`).

### Useful knobs

- `POLICY_FACTORY=<module>:<fn>` selects the policy (unset &rarr; `RandomPolicy`).
- `TASK` (e.g. `TurnOnSinkFaucet`, `OpenDrawer`, `PnPCounterToCab`), `SEED` make runs reproducible.
- `PORT` changes the browser port; `VIEW_RES` / `VIDEO_RES` size the live/saved video; `RT_HZ` the real-time rate.

### Layout

```
packages/simulation/robocasa
  Dockerfile              # ROCm base -> robosuite(master)+robocasa stack + sim_robocasa harness
  config.yaml             # ryzers manifest (gpu, EGL env, /sim_outputs + assets mounts, POLICY_FACTORY)
  test.py                 # import sign-of-life (no env build, no render, no model)
  scripts/setup_robocasa.sh  # runtime fetch of the ~10GB kitchen assets into the mounted volume
  examples/template_policy.py # COPY-ME annotated Policy skeleton (docs, not baked)
  lib/sim_robocasa/       # harness (installed to /opt/sim, on PYTHONPATH)
    policy.py             #   Policy ABC + load_policy() (POLICY_FACTORY seam)
    random_policy.py      #   built-in RandomPolicy (default, no weights)
    robocasa_env.py       #   vendored RoboCasa glue (create_env, render_obs, tasks, horizons)
    scene.py              #   per-task scene wrapper (reset/observe/step/success)
    rollout.py            #   model-agnostic chunk-replay episode loop
    render.py             #   MJPEG/JPEG, composed view, banner, even-dim MP4
    sanity.py             #   headless RandomPolicy rollout -> MP4
    interactive_server.py #   command-driven HTTP/MJPEG demo (chunk-replay)
    interactive_server_rt.py  # real-time HTTP/MJPEG demo (async planner + HOLD)
  demos/                  # demo_sim_sanity.sh, demo_interactive.sh, demo_interactive_rt.sh
```

Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
