# RoboLab-AMD — G4 Closed-Loop Report (BananaInBowl × Cosmos3-Nano-Policy-DROID)

Outcome of the G4 gate from `PILOT_PLAN.md`: an **AMD-native closed-loop bridge that runs the
NVIDIA Cosmos3-Nano-Policy-DROID policy end-to-end on ROCm** (Strix Halo, `gfx1151`) against
the real YCB BananaInBowl scene, with correct observation/action I/O and DROID-aligned
cameras. The policy performs **genuine, repeatable grasp attempts** but did **not** complete a
successful pick-and-place in the seeds run (0/5). Per the plan's own decision rule this is the
**"G1–G3 pass, G4 at chance → partial"** branch: the infra is a valid bridge; the residual gap
is a real-policy-in-sim domain gap, documented below.

---

## 1. Setup

- **Policy weights**: NVIDIA **Cosmos3-Nano-Policy-DROID** (16B world-action MoT), served in the
  `cosmos3:latest` image as an OpenPI-style WebSocket policy server on `:8000` with ROCm patches
  (MIOpen conv3d backend disabled to avoid GPU hangs; SDPA attention backend registered;
  guardrails disabled to drop the `nltk` dependency).
- **Client**: the sim image (`capx:robosuite-rocm`) runs `sim_robolab.closedloop` with
  `POLICY_FACTORY=cosmos3_websocket_policy:build_policy`, connecting over the vendored
  OpenPI msgpack+WebSocket glue (`vendor/openpi_client`, `vendor/openpi_server`; keepalive
  pings disabled — a single inference is ~26 s on `gfx1151`, longer than the default ping
  timeout).
- **Scene**: real YCB `011_banana` + `024_bowl` meshes, JOINT_POSITION control (kp=200),
  `object_in_container` success predicate (G3).
- **Inference latency**: mean **~26.2 s / chunk** (32 chunks per 500-step episode).

## 2. Bugs found and fixed (the substantive work)

### 2.1 Gripper proprio normalization (root cause of "never grasps")
`robolab_env.gripper_openness()` computed `joint_pos[0] / actuator_max[0]`. That formula was
inherited from the RoboCasa reference, which uses a **Panda** gripper (linear finger → clean
[0,1]). RoboLab uses a **Robotiq85** gripper (revolute driver joint `gripper0_right_finger_joint`,
range `[0, 0.8]`), so the ratio was an **unclamped, sign-inverted closedness** that read **3.100**
at the open pose. The adapter then sent `1 − 3.100 = −2.100` as the DROID `gripper_position`
(expected 0–1), i.e. a wildly out-of-distribution proprio every step. Fed that, the policy pinned
its gripper output at ~−0.9 → mapped to sim **−1 (fully open)** on every chunk. The gripper
**never closed**.

Fix (`gripper_openness`): return a **clamped openness = 1 − closedness ∈ [0,1]** (empirically
validated with `agent_scripts/probe_gripper.py`: driver joint ~0.0 open, ~0.65 closed,
`actuator_max=0.8`). After the fix the proprio reads `1.000` open and drops toward `~0.4` as the
gripper closes on the banana, and the policy's raw gripper output swings positive to **0.6–0.96**
(commands a real close).

### 2.2 Stale deployed package
The `robolab_pkg` mounted into the sim container had drifted from the source tree (the closed
loop saw `openness=3.100` where a fresh probe on the same mount saw `0.0`). Redeploying the full
`sim_robolab` package as one tarball resolved the discrepancy and is the reproducible path going
forward.

## 3. Camera rig alignment to DROID

DROID uses a wrist-mounted ZED-Mini + two adjustable exterior ZED-2 cameras (~69° vertical FoV);
the exteriors are deliberately varied across the dataset, so the **wrist camera is the
consistent, high-leverage view**. The pilot shipped robosuite stand-ins
(`robot0_eye_in_hand`, `agentview`, `sideview`) where the wrist cam framed the Robotiq fingers as
a large **near-field black occluding mass** (see before/after artifacts).

Fix (`robolab_env.apply_droid_cameras`, gated by `ROBOLAB_DROID_CAMS=1`, applied per-render so it
survives hard resets):
- **Wrist**: recomputed each frame as a world-frame **over-the-gripper look-at** — float the cam
  above-and-behind the grasp site (`WRIST_BACK=0.15`, `WRIST_UP=0.22`) looking down at it,
  expressed back into the hand body frame. Now frames the fingers-from-above with the banana/bowl
  visible in the workspace below.
- **Exteriors**: FoV widened 45° → **70°** (`EXT_FOVY`) toward the DROID ZED-2 distribution.

Verified cheaply with static renders (`agent_scripts/probe_cameras.py`) before spending
policy-run time.

## 4. Results

| Run | Cameras | Gripper proprio | Result |
|---|---|---|---|
| Original | stand-in | broken (`3.100`) | gripper never closes; banana untouched |
| Fix 2.1 (+redeploy) | stand-in | fixed | policy commands grasps; gripper closes; still misses |
| Fix 3 (DROID cams) | DROID-aligned | fixed | firm grasps + gripper closing (openness 1.0→~0.4); **0/5** |

Live telemetry after the fixes (from `COSMOS_DEBUG=1`): joint targets track the arm into the
banana region; `grip_raw` spikes to 0.6–0.96 (firm close); `in_openness` drops 1.0→~0.4 (real
closing). The failure mode is a **systematic spatial offset** — the gripper closes just
beside/behind the banana on **every** seed (placement-invariant), so more seeds do not flip it.

## 5. Diagnosis of the residual gap

The miss being systematic (not placement-dependent) points away from cameras/plumbing (now
correct) toward one of:
1. **Action-space / joint-convention offset** between DROID's Franka and robosuite's Panda — a
   small constant joint-zero offset yields exactly "reaches the region, closes slightly off".
2. **Exterior camera *angles***: FoV was matched but positions are robosuite front/90°-side, not
   DROID's ~45° over-shoulder pair.
3. **Visual domain gap**: a real-image-trained 16B policy grounding on synthetic white-table
   renders — would need domain randomization/realism or policy fine-tuning (large scope).

## 6. Reproduce

Server (remote, ROCm):
```
bash agent_scripts/run_cosmos3_server.sh          # cosmos3:latest policy server on :8000
```
Camera check (static, cheap):
```
ROBOLAB_DROID_CAMS=1 bash agent_scripts/run_probe_cameras.sh   # -> /out/cam_probe/stitched_*.png
```
Closed loop (DROID cameras, fixed gripper I/O):
```
SEED=0 NUM_EPISODES=5 STEPS=500 REPLAN_STEPS=16 \
  GRIPPER_INVERT=0 COSMOS_DEBUG=1 ROBOLAB_DROID_CAMS=1 STOP_ON_SUCCESS=1 \
  bash agent_scripts/run_robolab_closedloop.sh    # -> /out/closedloop_*.mp4 + summary.json
```

## 7. Artifacts (laptop, curated small)

- `artifacts/robolab/g4/wrist_before.png` — wrist view before the camera fix (black-blob occlusion).
- `artifacts/robolab/g4/wrist_after.png` — wrist view after the over-gripper look-at fix.
- `artifacts/robolab/g4/grasp_attempt.png` — 3-view during a (failed) grasp attempt with fixes on.

Weights, full images and MP4s stay on the remote (rules 3–4).
