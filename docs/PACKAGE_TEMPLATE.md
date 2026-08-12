# Package Template

The canonical layout and README shape for every policy, world-model, video-gen, and
simulation package on the benchmark branch. `packages/vla/molmoact2` is the reference
implementation. New and existing packages should match it so the whole tree reads the
same way once it is upstreamed.

Keep it minimal. State what the package is, how to build it, and how to reproduce each
demo. Do not narrate what was tried, what failed, or the porting history.

## Model package layout (vla, wam, vidgen)

```
packages/<class>/<name>/
  README.md          # sections below, one embedded visual per capability
  config.yaml        # ryzers build/run options (see docs/packages.md)
  Dockerfile         # ARG BASE_IMAGE ... ; CMD ["python","/ryzers/test.py"]
  test.py            # weightless ROCm + deps sign-of-life, exits non-zero on failure
  scripts/           # python entrypoints + download_*.sh helpers
  demos/             # demo_*.sh, one per capability in the README table
  assets/            # small gifs/pngs embedded in the README (keep well under a few MB each)
  docs/              # UPSTREAM_PIN.commit.txt and any deep-dive notes
  adapters/          # sim Policy-seam adapters (only if the package runs closed-loop)
  patches/           # upstream patches applied at build time (only if needed)
  RUNTIME_OPTIMIZATION.md   # WAM only: the single allowed file beyond this template
```

The only content a WAM package may add beyond the reference layout is optimization work:
`RUNTIME_OPTIMIZATION.md` and, if needed, an `experiments/` folder. Nothing else.

## Simulation package layout

```
packages/simulation/<name>/
  README.md          # what the sim is, build, one worked model example + one visual
  config.yaml
  Dockerfile
  test.py            # sim imports + a headless render sign-of-life
  scripts/
  demos/
  examples/
  lib/               # the model-agnostic Policy seam the model packages chain on
  assets/            # one demo gif of a paired model running in this sim
  patches/           # only if needed
```

## README skeleton: model packages

````markdown
### <ModelName>

This package runs [<ModelName>](<upstream-url>) on AMD Ryzen AI Max+ 395 (Strix Halo,
gfx1151) under ROCm <version>. <One sentence naming the model class and what the demos show.>

### Build

```sh
ryzers build <name>                        # standalone: model sign-of-life + non-sim demos
ryzers build simulation/<sim> <name>       # chain on a simulator base for closed-loop rollouts
ryzers run                                 # test.py: ROCm torch + GPU + deps check
```

Artifacts are written to `workspace/<name>/outputs`. Set `HF_TOKEN` for faster or gated
downloads. Weights are fetched on the first model run.

### Demos

| Demo | Base | What it does |
|---|---|---|
| `demos/demo_smoke.sh` | plain | Load checkpoint, one prediction, cold/steady latency. |
| `demos/demo_openloop.sh` | plain | Replay observations, overlay predicted vs ground-truth actions. |
| `demos/demo_videogen.sh` | plain | Imagine future frames from the first observation (WAM only). |
| `demos/demo_closedloop_<sim>.sh` | `<sim>` | Closed-loop rollouts and success rate. |
| `demos/demo_interactive_<sim>.sh` | `<sim>` | Interactive control over HTTP. |

### Open-loop replay

<One line with the number.>

<p align="center">
  <img src="assets/openloop.png" width="700">
  <br><em>Ground truth (solid) vs predicted (dashed) actions.</em>
</p>

### Video imagination

<WAM only. One line. Ground truth on the left, imagined future on the right.>

<p align="center">
  <img src="assets/imagination.gif" width="600">
  <br><em>Ground truth vs imagined future.</em>
</p>

### Closed-loop <sim>

<One line with the success rate.>

<p align="center">
  <img src="assets/closedloop.gif" width="420">
  <br><em>Closed-loop rollout in <sim>.</em>
</p>

### Interactive demo

Open `http://localhost:<port>` after:

```sh
ryzers run --name <name>-<sim> /ryzers/demos/demo_interactive_<sim>.sh
```

<p align="center">
  <img src="assets/interactive.gif" width="480">
  <br><em>Interactive control of <sim>.</em>
</p>

### Useful knobs

- <env var>: <what it does>

### References

- Upstream: <url> (pinned in `docs/UPSTREAM_PIN.commit.txt`)
- Model: <hf-url>
- Datasets: <hf-url>

Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
````

## README skeleton: simulation packages

````markdown
### <SimName>

This package provides the <SimName> simulation base (<backend>, e.g. MuJoCo/EGL or
SAPIEN/Vulkan) as a Ryzer on AMD Ryzen AI Max+ 395 (Strix Halo, gfx1151). It exposes a
model-agnostic `Policy` seam that WAM and VLA packages chain on for closed-loop and
interactive rollouts.

### Build

```sh
ryzers build simulation/<name>
ryzers run     # test.py: sim import + headless render sign-of-life
```

### Example: <SimName> with <Model>

```sh
ryzers build simulation/<name> <model>
ryzers run --name <model>-<name> /ryzers/demos/demo_closedloop_<name>.sh
```

<p align="center">
  <img src="assets/<name>_<model>.gif" width="480">
  <br><em><Model> closed-loop rollout in <SimName>.</em>
</p>

### References

- Upstream: <url> (pinned in `docs/UPSTREAM_PIN.commit.txt`)

Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
````

## Authoring rules

- Write plainly and briefly. No em-dashes. No "we tried", "worked", "did not work", or
  porting narrative in the base README. Reproduction steps and results only.
- Every capability section embeds at least one visual from `assets/`.
- When a ground-truth reference exists, show ground truth on the left and the prediction
  on the right, or overlay them on one plot for numeric data.
- For a purely generative result with no reference, show the generated output alone. If a
  single reference image drives the generation, keep it on the left and the output on the right.
- Keep every asset small. Prefer short looping gifs and downsampled plots. Large source
  videos stay on the remote, never in the package.
- Every package ends with the AMD copyright line and the MIT SPDX identifier.
- Pin the upstream commit in `docs/UPSTREAM_PIN.commit.txt` and reference it, so the build
  is reproducible and no upstream binaries are vendored into this repo.

## Conformance checklist

- [ ] Folder layout matches the template for its class.
- [ ] README follows the skeleton, section order, and tone.
- [ ] One embedded visual per capability section, all under `assets/`.
- [ ] `config.yaml`, `Dockerfile`, `test.py` present; `test.py` is weightless and exits non-zero on failure.
- [ ] Upstream pinned in `docs/UPSTREAM_PIN.commit.txt`; no vendored upstream weights or binaries.
- [ ] WAM extras limited to `RUNTIME_OPTIMIZATION.md` (+ optional `experiments/`).
- [ ] No AMD-internal host, cluster, or credential details anywhere in the package.
