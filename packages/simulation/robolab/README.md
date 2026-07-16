### simulation/robolab — RoboLab-AMD

Model-agnostic, **AMD-native reimplementation** of NVIDIA's [RoboLab-120](https://github.com/NVlabs/RoboLab)
closed-loop benchmark on the open [robosuite](https://github.com/ARISE-Initiative/robosuite)
(v1.5.1) / MuJoCo backend, for AMD Ryzen AI Max+ 395 (Strix Halo, `gfx1151`) under ROCm 7.2.2
(headless EGL).

Upstream RoboLab is **NVIDIA-only** (Isaac Sim/Lab + Omniverse RTX + Warp/PhysX GPU — closed
binaries; see `../../../docs/nano-world-model/ROBOLAB_AMD_*`). This package rebuilds the
benchmark's *task semantics* — DROID Franka + Robotiq 2F-85 embodiment, 3-camera rig, scene
layout, and success predicates — on a ROCm-friendly stack we already run on `gfx1151`.

> **It does NOT reproduce upstream's official numbers** (different physics engine and
> renderer). It IS a genuine AMD-native, model-agnostic closed-loop bridge that any policy
> (Cosmos3-Nano-Policy, X-WAM, ...) can be scored against on our hardware.

Status: **pilot** — see [`docs/PILOT_PLAN.md`](docs/PILOT_PLAN.md) for the task spec and the
four go/no-go gates.
- **G1 render+embody — PASS**: the custom `BananaInBowl` env builds and renders the 3-camera view
  headless on gfx1151 under `JOINT_POSITION` control with no hang.
- **G2 action bridge — PASS**: scripted joint chunks track to <0.04 rad (kp=200).
- **G3 predicate — PASS**: `object_in_container` fires on a real grasp+place and rejects 3 negatives.
- **G4 policy — PARTIAL**: the real **Cosmos3-Nano-Policy-DROID** (16B) drives BananaInBowl
  end-to-end on ROCm with correct I/O + DROID-aligned cameras and makes genuine grasp attempts,
  but did not complete a pick-and-place (0/5; systematic, placement-invariant grasp offset). Full
  account in [`docs/G4_REPORT.md`](docs/G4_REPORT.md).

### Build

```sh
ryzers build robolab --name sim-robolab
ryzers run --name sim-robolab                                  # test.py: import sign-of-life
ryzers run --name sim-robolab /ryzers/demos/demo_sim_sanity.sh # RandomPolicy rollout -> MP4
```

`robosuite==1.5.1` is installed from PyPI (code only). The pilot task `BananaInBowl` uses the
**real YCB `011_banana` + `024_bowl` meshes — the same source assets upstream RoboLab uses** —
fetched from the open YCB S3 bucket and convex-decomposed (coacd, so the bowl keeps its cavity)
into robosuite MJCFs at runtime into `workspace/simulation-robolab/assets` (never baked; rules
3 & 8). Set `ROBOLAB_USE_YCB=0` to fall back to primitives (no network). Sanity videos land
under `workspace/simulation-robolab/outputs` (mounted at `/sim_outputs`).

### Plug in your own policy (model entrypoint)

A policy implements `sim_robolab.Policy`:

```python
class Policy:
    replan_steps = 16     # env steps executed per predicted chunk before replanning
    num_steps_wait = 0
    def reset(self, instruction): ...
    def predict_action_chunk(self, obs, instruction) -> np.ndarray:  # [T, 8] joint targets + gripper
```

`obs` is the harness observation dict: `obs["video"]` (`[3,H,W,3]` float32 in `[-1,1]`, the 3
cameras `robot0_eye_in_hand`/`agentview`/`sideview`), `obs["qpos"]` (`[7]` arm joint angles),
`obs["proprios"]` (`[16]` eef pose + gripper + zeros), `obs["view"]` (stitched display frame),
and `obs["instruction"]`. The return is `[T, 8]` = **7 absolute Franka joint targets (rad) +
gripper** — matching Cosmos3-Nano-Policy-DROID's joint-position action head. The harness maps
absolute targets to the `JointPositionController`'s per-step delta and owns the env, rendering
and the episode loop. Copy [`examples/template_policy.py`](examples/template_policy.py) to wire
a model; select it at runtime with `POLICY_FACTORY=<module>:build_policy` (unset → RandomPolicy).

### Run a real policy: Cosmos3-Nano-Policy-DROID (G4)

The reference integration drives the sim with NVIDIA's **Cosmos3-Nano-Policy-DROID** (16B
world-action MoT). It is **model-free in this image**: the 16B policy runs in its own
`cosmos3` image as an OpenPI-style WebSocket policy server, and this sim image talks to it over
the vendored msgpack+WebSocket glue ([`vendor/openpi_client`](vendor/openpi_client),
[`vendor/openpi_server`](vendor/openpi_server); see [`vendor/VENDOR.md`](vendor/VENDOR.md)).
The client adapter is [`examples/cosmos3_websocket_policy.py`](examples/cosmos3_websocket_policy.py)
and the closed-loop, multi-seed eval harness is
[`lib/sim_robolab/closedloop.py`](lib/sim_robolab/closedloop.py).

Client (this image), against a server already on `:8000`:

```sh
POLICY_FACTORY=cosmos3_websocket_policy:build_policy \
POLICY_SERVER_HOST=127.0.0.1 POLICY_SERVER_PORT=8000 \
ROBOLAB_USE_YCB=1 ROBOLAB_DROID_CAMS=1 \
SEED=0 NUM_EPISODES=5 STEPS=500 REPLAN_STEPS=16 STOP_ON_SUCCESS=1 \
  python -m sim_robolab.closedloop            # -> /sim_outputs/closedloop_*.mp4 + summary.json
```

**What made it work (all reusable, in this package):**
- **Gripper proprio fix** (`robolab_env.gripper_openness`): the Robotiq85 driver joint
  (`gripper0_right_finger_joint`, range `[0,0.8]`) needs a *clamped* `openness = 1 − joint/0.8 ∈
  [0,1]`. The prior Panda-derived formula returned `3.100`, so the adapter fed the policy an
  out-of-distribution gripper state and it never closed. After the fix `grip_raw` swings to
  0.6–0.96 (firm close) and openness drops 1.0→~0.4 (real grasp).
- **DROID camera rig** (`robolab_env.apply_droid_cameras`, `ROBOLAB_DROID_CAMS=1`): the wrist cam
  is recomputed each frame as a world-frame over-the-gripper look-at (tunables `WRIST_BACK`,
  `WRIST_UP`, `WRIST_FOVY`) so the fingers frame from above with the workspace below (was a black
  occluding blob); exterior FoV widened 45°→70° (`EXT_FOVY`) toward DROID's ZED-2 distribution.
- **Gripper convention**: DROID `gripper_position` is `0=open..1=closed`; the sim uses `−1=open..
  +1=close`. The adapter maps `2g−1` (flip with `GRIPPER_INVERT=1` if ever needed — not needed
  here). Set `COSMOS_DEBUG=1` to log per-chunk joint/gripper telemetry.

**Server-side requirements** (in the `cosmos3` image, for ROCm/gfx1151): disable the MIOpen
conv3d backend (avoids a GPU hang), register an SDPA attention backend, disable guardrails (drops
the `nltk` dep), and disable WebSocket keepalive pings (a single inference is ~26 s, longer than
the default ping timeout).

**Result**: correct closed-loop behavior and genuine grasp attempts, 0/5 success — a systematic,
placement-invariant grasp offset remains (likely a DROID↔robosuite joint-convention offset and/or
the real-policy-in-sim visual domain gap). See [`docs/G4_REPORT.md`](docs/G4_REPORT.md) for the
full diagnosis and next levers.

### Layout

```
packages/simulation/robolab
  Dockerfile              # ROCm base -> robosuite 1.5.1 + MuJoCo + sim_robolab harness
  config.yaml             # ryzers manifest (gpu, EGL env, /sim_outputs mount, POLICY_FACTORY)
  test.py                 # import + JOINT_POSITION controller sign-of-life (no env build/render)
  docs/PILOT_PLAN.md      # task spec + G1..G4 go/no-go gates
  docs/G4_REPORT.md       # Cosmos3-Nano-Policy-DROID closed-loop report (bugs fixed, cameras, results)
  scripts/setup_robolab.sh   # runtime asset fetch (no-op for the procedural pilot)
  examples/template_policy.py # COPY-ME annotated Policy skeleton (joint-space contract)
  examples/cosmos3_websocket_policy.py # Cosmos3-Nano-Policy-DROID client adapter (G4 seam)
  vendor/openpi_client, vendor/openpi_server # minimal OpenPI msgpack+WebSocket glue (see VENDOR.md)
  lib/sim_robolab/        # harness (installed to /opt/sim, on PYTHONPATH)
    policy.py             #   Policy ABC + load_policy() (POLICY_FACTORY seam)
    random_policy.py      #   built-in joint-space RandomPolicy (default, no weights)
    robolab_env.py        #   create_env (DROID embodiment + 3 cams + JOINT_POSITION), render_obs
    tasks/banana_in_bowl.py   #   custom robosuite ManipulationEnv (TableArena + YCB banana + bowl)
    ycb_to_mjcf.py        #   fetch YCB meshes + coacd decomposition -> robosuite MJCF
    predicates.py         #   object_in_container (MuJoCo-state reimpl of RoboLab conditional)
    scene.py              #   reset/observe/step/success wrapper
    rollout.py            #   model-agnostic chunk-replay episode loop
    closedloop.py         #   G4 multi-seed closed-loop eval (success rate + MP4 + latency)
    render.py             #   composed 3-view, banner, even-dim MP4
    sanity.py             #   headless RandomPolicy rollout -> MP4
  demos/demo_sim_sanity.sh
```

Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
