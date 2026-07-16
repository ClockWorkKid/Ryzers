# RoboLab-AMD — Pilot Plan (BananaInBowl)

AMD-native reimplementation of the NVIDIA **RoboLab-120** closed-loop benchmark on our
ROCm MuJoCo/robosuite stack (Strix Halo, `gfx1151`, ROCm 7.2.2). Upstream RoboLab is
NVIDIA-only (Isaac Sim/Lab + Omniverse RTX + Warp/PhysX GPU — closed binaries, see
`../../../../docs/nano-world-model/ROBOLAB_AMD_*.md`). This package rebuilds the benchmark
*task semantics* (scene layout, DROID embodiment, camera rig, success predicates) on the
open, ROCm-friendly robosuite/MuJoCo backend we already run headless on `gfx1151`.

> **It will NOT reproduce upstream's official numbers** — different physics engine
> (MuJoCo vs PhysX/Warp) and renderer (MuJoCo EGL vs Omniverse RTX). It IS a genuine
> AMD-native, model-agnostic closed-loop bridge that any policy (Cosmos3-Nano-Policy,
> X-WAM, ...) can be scored against on our hardware.

The whole effort is **gated on a 1–2 task pilot** (`BananaInBowlTask`). We build the
minimum to clear four gates, measure them, and only then decide whether to scale to the
full 120.

---

## 1. Pilot task spec (from upstream `banana_in_bowl_task.py`)

| Field | Upstream value | AMD-native mapping |
|---|---|---|
| scene | `banana_bowl.usda` (banana + bowl + table); objects are YCB `011_banana` + `024_bowl` (per upstream `object_catalog.json`) | robosuite `TableArena` + the **same** YCB `011_banana` + `024_bowl` meshes, fetched from the open YCB S3 bucket and coacd-decomposed to MJCF at runtime (`ycb_to_mjcf.py`); primitive fallback if `ROBOLAB_USE_YCB=0` |
| robot | `DroidCfg` = Franka + **Robotiq 2F-85**, joint-pos action, high PD (400/80), gravity-disabled arm | robosuite `Panda` + `Robotiq85Gripper`, `JOINT_POSITION` controller |
| cameras | wrist (on gripper) + over-shoulder left + over-shoulder right (`WRIST_LEFT_RIGHT` preset) | `robot0_eye_in_hand` + 2 scene cams `agentview_left`/`agentview_right` posed to approximate the DROID rig |
| instruction | default: "Pick up the banana and place it in the bowl" | identical (passed to policy) |
| episode_length_s | 50 | mapped to step budget at control Hz (see §4) |
| success | `object_in_container(object="banana", container="bowl", gripper_name="gripper", tolerance=0.0, require_contact_with=True, require_gripper_detached=True)` | reimplemented geometrically in MuJoCo (§3) |
| subtask | `pick_and_place(object=["banana"], container="bowl")` | same predicate, progress-tracked |

**Success predicate semantics** (`object_in_container` with these params): banana is
inside the bowl's opening footprint **and** the banana was in contact with the gripper at
some point **and** the gripper is now detached from the banana (i.e. the arm actually
grasped, transported, and released it — not knocked it in).

---

## 2. Architecture (mirrors `simulation/robocasa`)

Model-agnostic simulator base image; policy chains on top and is selected at runtime via
`POLICY_FACTORY` (identical seam to robocasa/libero/simplerenv, so Cosmos3 / X-WAM adapters
are near-drop-in).

```
packages/simulation/robolab/
  Dockerfile              # ROCm base -> robosuite(master) + MuJoCo + sim_robolab harness
  config.yaml             # ryzers manifest (gpu, EGL env, /sim_outputs + assets mounts, POLICY_FACTORY)
  test.py                 # import sign-of-life (no env build / render / model)
  scripts/setup_robolab.sh   # runtime fetch+convert of YCB banana/bowl meshes (rule 8: not baked)
  examples/template_policy.py
  lib/sim_robolab/
    policy.py             # Policy ABC + load_policy() (POLICY_FACTORY seam)
    random_policy.py      # built-in RandomPolicy (joint-space, no weights)
    robolab_env.py        # create_env(): robosuite env + 3 DROID cams + JOINT_POSITION ctrl
    tasks/banana_in_bowl.py   # custom robosuite ManipulationEnv (TableArena + YCB banana + bowl)
    ycb_to_mjcf.py        # YCB fetch + coacd convex decomposition -> robosuite MJCF (+ meta)
    predicates.py         # object_in_container + helpers (MuJoCo-state reimpl of conditionals)
    scene.py              # reset/observe/step/success wrapper (mirrors robocasa scene.py)
    rollout.py            # model-agnostic chunk-replay episode loop
    render.py             # composed 3-view, banner, even-dim MP4 (reuse robocasa render.py)
    sanity.py             # headless RandomPolicy rollout -> MP4
  demos/demo_sim_sanity.sh
```

**Key deltas vs robocasa** (everything else is reused verbatim):
1. `robolab_env.py` builds a **custom tabletop robosuite env** (robosuite has no banana-in-bowl
   task) instead of `robosuite.make("PnP...")`.
2. Controller is **`JOINT_POSITION`** (Cosmos3-Nano-Policy-DROID emits absolute Franka joint
   targets), not OSC_POSE deltas — `scene.step()` feeds 7 joint targets + gripper directly.
3. `check_success` calls our `predicates.object_in_container`, not `env._check_success()`.
4. Cameras named/posed to the DROID rig, not RoboCasa's kitchen cams.

---

## 3. Success predicate (MuJoCo reimplementation)

`object_in_container(banana, bowl)` returns True iff **all**:
- **containment**: banana body COM is within the bowl's rim radius in XY and its Z is between
  the bowl base and rim (bowl AABB from its geom, banana COM from `sim.data.body_xpos`).
- **was_grasped**: at some step in the episode, a gripper finger geom had contact with the
  banana (`require_contact_with=True`) — tracked with a sticky flag over the episode.
- **gripper_detached**: no current contact between either gripper finger geom and the banana
  (`require_gripper_detached=True`).
- **settled**: banana near-stationary for K consecutive steps (avoids counting a mid-air frame).

All from `sim.data` (body/geom xpos + `sim.data.contact`) — no renderer needed. Validated in
G3 with scripted trajectories (true positive + 3 negatives: never-grasped, still-held, missed-bowl).

---

## 4. Action / proprio contract (for the Cosmos3 adapter, G4)

- **Action** (policy -> sim): Cosmos3-Nano-Policy-DROID chunk = absolute Franka **joint
  positions** (7) + gripper (1). `scene.step` sets the `JOINT_POSITION` controller targets
  directly (no IK). Gripper: binary open/close mapped to Robotiq85 actuator range.
- **Proprio** (sim -> policy): DROID proprio layout the Cosmos3 client expects (joint pos +
  gripper). Exact layout confirmed against the open-loop eval client before G4.
- **Video** (sim -> policy): 3 cams composed exactly as the open-loop eval expects
  (wrist top / left|right bottom). `replan_steps` = policy action-horizon.
- Control Hz + `episode_length_s`(50) -> `MAX_STEPS`; matched to robocasa's step-budget style.

---

## 5. Go / No-Go gates

| Gate | Criterion | Measurement |
|---|---|---|
| **G1 render+embody** — **PASS** | Franka + Robotiq 2F-85 + YCB banana + bowl + table build and render **headless on gfx1151**; 3 cameras produce RGB; composed view streams; no MIOpen/EGL hang | `demo_sim_sanity.sh` RandomPolicy rollout -> MP4 (`artifacts/robolab/banana_in_bowl_ycb*`) |
| **G2 action bridge** — **PASS** | A scripted **joint-position** chunk drives all 7 arm joints to commanded targets under `JOINT_POSITION` control; gripper opens/closes. Achieved: mean RMSE **0.021 rad**, max **0.037 rad** (kp=200, crit. damped) | `demo_ctrl_tracking.sh` (ScriptedJointPolicy + `tracking.py`) commanded-vs-achieved overlay (`artifacts/robolab/g2_tracking.png`) |
| **G3 predicate** — **PASS** | `object_in_container(banana,bowl)` fires on a genuine grasp+place (real YCB banana settled in real bowl, gripper detached) and stays False for 3 negatives, each isolating one gating branch: never_grasped (`was_grasped=0`), still_held (`detached=0`), missed_bowl (`contained=0`) | `predicate_check.py` real-asset unit test + annotated 4-panel (`artifacts/robolab/g3_predicate.png`). Added `protrusion_tol` (banana rests proud of the small bowl rim) |
| **G4 policy** — **PARTIAL** | Cosmos3-Nano-Policy-DROID drives BananaInBowl end-to-end on ROCm over the OpenPI/`POLICY_FACTORY` seam with correct I/O + DROID cameras; policy makes genuine grasp attempts (gripper closes) but 0/5 (systematic spatial offset, placement-invariant) | see `G4_REPORT.md`; matches the "G4 at chance → partial" decision branch below |

**Decision rule after pilot:**
- **G1–G3 pass, G4 above chance** -> GO: scale to a first RoboLab task cluster (the
  `banana_bowl` / `bagel_plate_banana_bowl` scene family shares assets).
- **G1–G3 pass, G4 at chance** -> partial: infra is a valid AMD-native bridge, but flag the
  sim2sim domain gap (physics/renderer) as the likely cause; decide whether to invest in
  visual/domain matching before scaling.
- **G1 or G2 fails** -> NO-GO on this backend; revisit (unlikely — robocasa already clears
  the equivalent render+control path on gfx1151).

---

## 6. Effort / risk

- **Reused (low risk):** ROCm robosuite/MuJoCo base + EGL headless render (proven by
  robocasa/libero on gfx1151), the entire `sim_*` harness pattern (policy seam, rollout,
  render, sanity, interactive servers), the `POLICY_FACTORY` runtime wiring.
- **New (the real work):** the custom robosuite `BananaInBowl` env (arena + banana/bowl
  meshes + 3 DROID cameras + `JOINT_POSITION` controller), the `object_in_container`
  predicate, and the Cosmos3 joint-space adapter.
- **Main risks:** (a) graspability tuning of the YCB meshes (friction/coacd hull fidelity —
  mesh sourcing itself is resolved: real YCB `011_banana`/`024_bowl` render textured on
  gfx1151, bowl cavity preserved via 55 coacd hulls); (b) camera-pose domain gap vs DROID
  real rig hurting policy success (G4); (c) `JOINT_POSITION` gain tuning for stable tracking.
  All are localized and testable per-gate (rule 2).

Assets are **fetched at runtime, never vendored** (rule 8). Weights/large videos stay on
remote; only small artifacts + this doc mirror to the laptop (rules 3–4).
