### simulation/libero-plus

Model-agnostic [LIBERO-Plus](https://github.com/sylvestf/LIBERO-plus) robustness-benchmark
simulator base image for AMD Ryzen AI Max+ 395 (Strix Halo, `gfx1151`) under ROCm 7.2.2.

**LIBERO-Plus** is a drop-in replacement for [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)
that expands the four suites into **10,030 perturbation instances** across **7 robustness
dimensions** — camera viewpoints, robot initial states, language instructions, light
conditions, background textures, sensor noise, objects layout — stratified into 5 difficulty
levels (L1–L5). Evaluation is identical to LIBERO except each task runs a single trial (its
init state encodes one perturbation config).

This package ships **only the simulator**: the LIBERO-Plus / MuJoCo / robosuite closed-loop
stack (headless EGL) plus a small harness (`sim_liberoplus`) that exposes a model-agnostic
`Policy` seam, generic closed-loop / interactive / sanity runners, and a loader for the
perturbation classification (dimension + difficulty per task). It contains **no** policy or
model code. Any VLA/WAM (VLA-JEPA, ...) is layered on top as a separate ryzers image and
selected at runtime — see *Plug in your own policy* below.

### Build

```sh
ryzers build libero-plus --name sim-libero-plus
ryzers run --name sim-libero-plus                       # test.py: ROCm + LIBERO-Plus sign-of-life
ryzers run --name sim-libero-plus /ryzers/demos/demo_sim_sanity.sh   # RandomPolicy rollout -> MP4
```

The build downloads the **~6.4 GB perturbation asset pack** (objects, textures, scenes) from
the upstream open-source HuggingFace dataset ([`Sylvest/LIBERO-plus`](https://huggingface.co/datasets/Sylvest/LIBERO-plus))
into the image, in a dedicated cached layer (never re-hosted here). Use
`demos/demo_download.sh` to (re)download the assets explicitly. Sanity videos land under
`workspace/simulation-libero-plus/outputs` (mounted at `/sim_outputs`).

### Plug in your own policy (model entrypoint)

The simulator is driven through one small interface, so any model connects the same way the
upstream LIBERO benchmark takes a pluggable policy. A policy implements
`sim_liberoplus.Policy`:

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

1. Build your ryzer **FROM** the sim base, installing your model under the base's torch+numpy
   pins (LIBERO-Plus needs `numpy 1.26.4`):
   ```sh
   ryzers build libero-plus <yourmodel> --name <yourmodel>-libero-plus
   ```
2. Ship an adapter implementing `Policy` — copy [`examples/template_policy.py`](examples/template_policy.py)
   (annotated skeleton) or see [`examples/vjepa_liberoplus_policy.py`](examples/vjepa_liberoplus_policy.py)
   (VLA-JEPA-shaped stub). VLA-JEPA is the first model ported: its full adapter + closed-loop
   robustness runner live in `vla/vlajepa/adapters/` and `vla/vlajepa/scripts/`.
3. Select it at runtime:
   ```sh
   POLICY_FACTORY=<module>:build_policy ryzers run --name <yourmodel>-libero-plus \
     /ryzers/demos/demo_interactive.sh          # http://localhost:8080
   ```

Unset `POLICY_FACTORY` &rarr; the built-in `RandomPolicy`.

### Perturbation classification

`sim_liberoplus.libero_env` exposes helpers over the upstream `task_classification.json`:

- `load_task_classification()` &rarr; `{suite: [{id, name, category, difficulty_level}, ...]}`
- `resolve_task_ids(task_suite, entries)` &rarr; map entries to 0-based task ids for a built suite
- `PERTURBATION_CATEGORIES` &rarr; the 7 dimension names

A chained closed-loop runner uses these to slice the benchmark by `CATEGORY` / `DIFFICULTY`
and report per-dimension robustness the way the LIBERO-Plus paper does.

### Demos

| Demo | What it does |
|---|---|
| `demos/demo_sim_sanity.sh` | Headless `RandomPolicy` rollout &rarr; `/sim_outputs` MP4 (no model). |
| `demos/demo_interactive.sh` | Command-driven browser demo over HTTP/MJPEG (`RandomPolicy` default). |
| `demos/demo_interactive_rt.sh` | Real-time browser demo (async planner; arm HOLDs while thinking). |
| `demos/demo_download.sh` | (Re)download the perturbation asset pack from HuggingFace. |

Chained policy packages add their own model-driven demos (e.g. VLA-JEPA's
`demo_interactive_liberoplus.sh`, `demo_closedloop_liberoplus.sh`).

### Useful knobs

- `POLICY_FACTORY=<module>:<fn>` selects the policy (unset &rarr; `RandomPolicy`).
- `SUITE`, `TASK_ID`, `SEED` make runs reproducible.
- `CATEGORY` (one of the 7 dimensions), `DIFFICULTY` (1–5), `MAX_TASKS` slice the benchmark
  for chained runners.
- `PORT` changes the browser port; `VIEW_RES` / `VIDEO_RES` size the live/saved video.

### Layout

```
packages/simulation/libero-plus
  Dockerfile              # ROCm base -> LIBERO-Plus stack + assets + sim_liberoplus harness
  config.yaml             # ryzers manifest (gpu, EGL env, /sim_outputs mount, POLICY_FACTORY)
  test.py                 # import sign-of-life + classification/suite counts (no render, no model)
  scripts/
    download_assets.sh    # (re)download the ~6.4 GB perturbation asset pack from HuggingFace
  examples/               # COPY-ME model-entrypoint adapters (docs, not baked)
    template_policy.py     #   annotated Policy skeleton
    vjepa_liberoplus_policy.py  # VLA-JEPA-shaped stub
  lib/sim_liberoplus/     # harness (installed to /opt/sim, on PYTHONPATH)
    policy.py             #   Policy ABC + load_policy() (POLICY_FACTORY seam)
    random_policy.py      #   built-in RandomPolicy (default, no weights)
    libero_env.py         #   LIBERO-Plus glue (env, image, dummy action, horizons, classification)
    scene.py              #   (suite, task_id) scene wrapper
    rollout.py            #   model-agnostic chunk-replay episode loop
    render.py             #   MJPEG/JPEG, composed view, banner, even-dim MP4
    sanity.py             #   headless RandomPolicy rollout -> MP4
    interactive_server.py #   command-driven HTTP/MJPEG demo (chunk-replay)
    interactive_server_rt.py  # real-time HTTP/MJPEG demo (async planner + HOLD)
  demos/                  # demo_sim_sanity.sh, demo_interactive.sh, demo_interactive_rt.sh, demo_download.sh
```

Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
