### simulation/mimicgen

Model-agnostic [MimicGen](https://github.com/NVlabs/mimicgen) (robosuite v1.4 / MuJoCo backend,
9 core manipulation tasks across the coffee / square / stack / stack_three families) simulator base
image for AMD Ryzen AI Max+ 395 (Strix Halo, `gfx1151`) under ROCm 7.2.2.

This package ships **only the simulator**: the robosuite / robomimic / NVlabs-mimicgen / MuJoCo
closed-loop stack (headless EGL, OSMesa software fallback) plus a small harness (`sim_mimicgen`)
that exposes a model-agnostic `Policy` seam and a generic closed-loop / sanity runner. It contains
**no** policy or model code. Any VLA/WAM (VERA, ...) is layered on top as a separate ryzers image
and selected at runtime — see *Plug in your own policy* below.

The eval harness itself (reset-to-demo, demo-state warmup, success tracking, per-view videos) is
VERA's upstream `vera.env_runner.MimicgenRunner` + websocket client, reused unchanged (rule 2.1).
We install **only the sim/eval half** (`vera[eval]` + robosuite/robomimic/mimicgen/mujoco), never
the planner/IDM (`[idm,video]` — diffusers/WAN/vggt), so this base stays lean; the model stack only
ever lands in the layer built on top.

### Build

```sh
ryzers build mimicgen --name sim-mimicgen                     # builds on the ROCm 7.2.2 base
ryzers run --name sim-mimicgen                                # test.py: ROCm + sim import sign-of-life
ryzers run --name sim-mimicgen /ryzers/scripts/download_mimicgen_datasets.sh stack_d0   # fetch a task hdf5
ryzers run --name sim-mimicgen /ryzers/demos/demo_sim_sanity.sh   # RandomPolicy rollout -> MP4
```

MimicGen "core" task hdf5 (env config + demo initial states) are fetched at run time from
`amandlek/mimicgen_datasets` into the mounted volume (`workspace/simulation-mimicgen/data`
→ `/sim_data`, rule 8), never baked. Sanity videos land under `workspace/simulation-mimicgen/outputs`
(mounted at `/sim_outputs`). Both are **distinct** container targets so that when a model is chained
on top, its `/outputs` + `/models` mounts are added without collision.

### Plug in your own policy (model entrypoint)

The simulator is driven through one small seam. A policy implements `sim_mimicgen.Policy` — the
runner calls it **per env step** (robosuite OSC_POSE: a 7-D delta `[dx,dy,dz, drx,dry,drz, gripper]`):

```python
class Policy(BasePolicy):
    view_keys = None       # image obs keys concatenated width-wise into obs.rgb (hint)
    context_frames = None  # frames warmed up before the policy first acts (hint)
    render_size = None      # per-view render edge px (hint)
    def reset(self): ...
    def predict_action(self, obs) -> PolicyOutput:  # obs.rgb is the (view-concat) frame
        ...
```

Chunked models (predict N actions, replay one per step) bridge that internally — the shipped
`sim_mimicgen.remote_policy.RemoteWebsocketPolicy` does exactly this for **any model served over the
websocket protocol** (obs→action-chunk), which is how VERA (WAN planner + Jacobian IDM) plugs in.

Connect a model in three steps (no edits to this package):

1. Build your ryzer **FROM** the sim base (chain build):
   ```sh
   ryzers build simulation/mimicgen <yourmodel> --name <yourmodel>-mimicgen
   ```
2. Provide a factory `build_policy() -> sim_mimicgen.Policy` — copy
   [`examples/template_policy.py`](examples/template_policy.py), or (for a heavy / dep-incompatible
   model) run it as a websocket policy server and reuse `sim_mimicgen.remote_policy:build_policy`.
3. Select it at runtime:
   ```sh
   POLICY_FACTORY=<module>:build_policy TASK=stack_d0 ryzers run --name <yourmodel>-mimicgen \
     /ryzers/demos/demo_closedloop.sh
   ```

Unset `POLICY_FACTORY` → the built-in `RandomPolicy` (sanity only, solves nothing).

### Demos

| Demo | What it does |
|---|---|
| `demos/demo_sim_sanity.sh` | Headless `RandomPolicy` rollout → `/sim_outputs` MP4 (no model). |
| `demos/demo_closedloop.sh` | Model-driven closed-loop for the `POLICY_FACTORY`-selected policy (in-process, or a running remote server via `POLICY_HOST/POLICY_PORT`). |

### Useful knobs

- `POLICY_FACTORY=<module>:<fn>` selects the policy (unset → `RandomPolicy`);
  `POLICY_HOST`/`POLICY_PORT` point the built-in remote policy at a running model server.
- `TASK` (`coffee_d0`/`square_d0`/`stack_d0`/`stack_three_d0`/…), `NUM_DEMOS`, `ROLLOUT_HORIZON`,
  `RENDER_SIZE`, `VIEWS` (comma-separated obs keys), `CONTEXT_FRAMES`, `SEED`.
- `HF_TOKEN` for faster/gated dataset downloads.

### Layout

```
packages/simulation/mimicgen
  Dockerfile              # ROCm base -> robosuite/robomimic/mimicgen/mujoco + vera[eval] + sim_mimicgen
  config.yaml             # ryzers manifest (gpu, EGL env, /sim_outputs + models mounts, POLICY_FACTORY)
  test.py                 # import sign-of-life (no env build, no render, no model)
  scripts/
    download_mimicgen_datasets.sh  # runtime fetch of the 9 core-task hdf5 (rule 8)
    strip_cuda_torch.py            # keep base ROCm torch (drop upstream CUDA pin)
    patch_sim_imports.py           # sim-side gfx1151 fix (disable ws-client keepalive)
    _hf_common.sh
  examples/template_policy.py      # COPY-ME annotated Policy skeleton (docs, not baked)
  lib/sim_mimicgen/       # harness (installed to /opt/sim, on PYTHONPATH)
    policy.py             #   Policy seam ABC + load_policy() (POLICY_FACTORY selector)
    random_policy.py      #   built-in RandomPolicy (default, no weights)
    remote_policy.py      #   RemoteWebsocketPolicy: drive from any obs->action-chunk model server
    rollout.py            #   model-agnostic closed-loop over the upstream MimicgenRunner
    sanity.py             #   CLI entry: `python -m sim_mimicgen.sanity`
  demos/                  # demo_sim_sanity.sh, demo_closedloop.sh
```

### References

- MimicGen: https://github.com/NVlabs/mimicgen · datasets: `amandlek/mimicgen_datasets`
- Harness/upstream: https://github.com/sizhe-li/VERA (pinned in the model layer's `docs/UPSTREAM_PIN.commit.txt`)
- robosuite: https://github.com/ARISE-Initiative/robosuite · robomimic: https://github.com/ARISE-Initiative/robomimic

Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
