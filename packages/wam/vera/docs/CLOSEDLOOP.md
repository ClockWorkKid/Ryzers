# VERA Closed-Loop Control + Interactive Viewer

Closed-loop control + interactive MJPEG viewer for both VERA embodiments on Strix Halo (Ryzen AI
Max+ 395, `gfx1151`, ROCm 7.2.2), using VERA's upstream server / controller / viewer unchanged
where possible (rule 2.1). Two embodiments, two viewer modes (recorded mp4 + live port-forward).

```
client  ──websocket :PORT──▶  start_vera_server (WAN planner + tracker + Jacobian IDM)
  │                                    │
  └─ steps the sim (gym-pusht /        └─▶ MJPEG viewer @ :VIS_PORT  ──▶ save_vis_video → mp4
     robosuite+MuJoCo, offscreen EGL)      (obs | dream+tracks | dream | Jacobian)
```

Packaging: VERA is now the **model layer** (`.[idm,video]` - planner + IDM), built on top of the
model-agnostic **`simulation/mimicgen`** base (robosuite / robomimic / NVlabs-mimicgen / MuJoCo +
EGL/OSMesa GL libs + the `sim_mimicgen` harness): `ryzers build simulation/mimicgen vera`. PushT
(`gym-pusht`, no MuJoCo) stays in the model image. For MimicGen the loop is split across the seam - 
VERA serves the WAN+IDM policy over the websocket protocol, and the sim base harness steps the env
against it (`demos/demo_closedloop_mimicgen.sh`, `POLICY_FACTORY=vera_mimicgen_policy:build_policy`).
`demos/demo_mimicgen.sh` still drives the same server through the upstream in-package controller
directly. Weights + frozen bases are fetched at runtime into the mounted caches (rule 8).

---

## PushT - `demos/demo_pusht.sh`

Lightest embodiment (DFoT planner + Jacobian IDM, no MuJoCo - `gym-pusht`). The demo brings up the
server (`--embodiment pusht --port 8820 --vis-port 8821`), waits for the policy port, runs the
headless `scripts/closedloop_pusht.py` client (ported from `examples/pusht_dfot_stack.ipynb`), then
dumps the viewer buffer to `pusht_vis.mp4`.

```sh
ryzers run --name vera /ryzers/demos/demo_pusht.sh
```

**Result**: 3 rollouts from replay state 3664 (horizon 200, render 252, seed 42) - 
**100% success, mean max-reward 100%, mean coverage 82%**. Videos: `pusht_vis.mp4` (viewer
dashboard) + per-episode `vis_env0.mp4` (dream+tracks overlay) + `policy_env0.mp4` + a summary JSON.

## MimicGen - `demos/demo_mimicgen.sh` (single task) + `demos/demo_mimicgen_suite.sh` (full suite)

robosuite/MuJoCo tasks driven by the omni **WAN-1.3B** video planner + the MimicGen **Jacobian
IDM**, all inference server-side. Uses the upstream controller `vera.controller.run_mimicgen_eval`
unchanged; MuJoCo renders offscreen via **EGL** with an **OSMesa** software fallback. Server:
`--embodiment mimicgen --port 8800 --vis-port 8801 --algo-config <cotracker-patched>
--sample-steps 40` (the shipped `algo_config.yaml` default; `--sample-steps 10` is a fast
sign-of-life setting only - it under-denoises and does not complete tasks).

**Full task range.** The MimicGen IDM was trained task-balanced over the nine core-task shards, so
all four object families run closed-loop: **coffee** (`coffee_d0/d1`), **square** nut-assembly
(`square_d0/d1/d2`), **stack** (`stack_d0/d1`), **three-piece stack** (`stack_three_d0/d1`).
`scripts/download_mimicgen_datasets.sh` fetches the nine `*.hdf5` from `amandlek/mimicgen_datasets`
at runtime (rule 8); `demo_mimicgen.sh` is parametrized by `TASK`, and `demo_mimicgen_suite.sh`
runs the whole set against **one warm policy server** (cold start paid once) with a per-task
EGL→OSMesa retry.

```sh
# verified success (upstream defaults):
TASK=stack_d0 NUM_DEMOS=1 ROLLOUT_HORIZON=700 SAMPLE_STEPS=40 ryzers run --name vera /ryzers/demos/demo_mimicgen.sh
# fast sign-of-life (pipeline check only, does not complete tasks):
TASK=stack_d0 NUM_DEMOS=1 ROLLOUT_HORIZON=100 SAMPLE_STEPS=10 ryzers run --name vera /ryzers/demos/demo_mimicgen.sh
NUM_DEMOS=1 ROLLOUT_HORIZON=700 SAMPLE_STEPS=40 ryzers run --name vera /ryzers/demos/demo_mimicgen_suite.sh  # all 9
```

**Verified result (upstream defaults - `stack_d0`, `H=700`, `sample_steps=40`, 1 demo).** The task
**succeeds: 1/1 = 100 %, max reward 1.000**, success latched at env step ~491 (chunk 50). This is
the canonical `examples/mimicgen_stack.ipynb` configuration (`ROLLOUT_HORIZON=700`, yaml
`sample_steps=40`, `action_scale=1.0`, `render_size=128`), reproduced end-to-end on `gfx1151`. The
earlier all-zero passes were **settings, not a port defect**: `H=100-200` truncated every episode
before completion, and `sample_steps=10` under-denoised the plan. Videos land in
`stack_d0_default/` (viewer dashboard + `agentview`/`obs` episode mp4s).

**Result (sign-of-life, `NUM_DEMOS=1 ROLLOUT_HORIZON=100`, one warm server, ~5.2 h wall):** every
task loads, steps closed-loop, and renders end-to-end - each server inference dreams a WAN rollout,
runs the multiview (agentview + eye-in-hand) **cotracker** motion tracker + `draw_pts_gpu` overlay,
and the Jacobian IDM commits `H=10` actions to the robosuite env. **8/9 tasks completed `rc=0`**
with viewer clips; only `square_d2` failed, on an intermittent >600 s keepalive stall (see notes).
Success rate is 0/1 across the board - expected and uninformative at this depth (100 steps ≪ the
task horizons); this pass validates the *pipeline* per task, not policy skill (a real success
measurement needs many demos × full horizon - days on one gfx1151, see runtime section).

| family | tasks | rc | clips |
|---|---|---|---|
| coffee | `coffee_d0`, `coffee_d1` | 0, 0 | 4 each |
| square | `square_d0`, `square_d1` | 0, 0 | 4 each |
| square | `square_d2` | **1** | - (keepalive stall, re-run candidate) |
| stack | `stack_d0`, `stack_d1` | 0, 0 | 4 each |
| stack_three | `stack_three_d0`, `stack_three_d1` | 0, 0 | 4 each |

Each `rc=0` task writes `{task}_vis.mp4` (viewer dashboard) + `agentview_image` / `eye_in_hand` /
`obs` episode mp4s (~0.4-0.5 MB/task). cotracker (`scaled_offline.pth`, ~101 MB) is pulled once via
`torch.hub` into the mounted cache (rule 8).

**Rendering note.** EGL occasionally hangs on `gfx1151` for a given env/run (surfacing as a
`libGLU.so.0` warning at env-close). The suite's EGL→OSMesa retry recovers it (`square_d0/d1` passed
on the OSMesa retry); rendering cost is negligible next to the ~142 s denoise, so preferring OSMesa
outright is a safe robustness lever. Adding `libglu1-mesa` at the next rebuild silences the cosmetic
warning.

**WebSocket keepalive (fixed - port fix #5).** The single biggest cause of lost long runs was the
WebSocket keepalive: an occasional chunk stalls well past the 600 s ping timeout on `gfx1151`
(hipblaslt→unfused-cublas fallbacks + memory pressure), which dropped the connection mid-denoise
(`sent 1011 keepalive ping timeout`) and killed the whole episode - and the EGL→OSMesa retry then
could not reconnect because the single-threaded server was still finishing the stalled chunk
(handshake `TimeoutError`). Keepalive is now **disabled on both client and server** so a merely-slow
forward pass never drops the link (the server does finish the chunk; a truly dead server still
surfaces via TCP close). The verified `stack_d0` success rode through **two anomalous stalls
(~1462 s and ~614 s)** that both exceed the old 600 s timeout - without this fix that exact rollout
would have died at chunk 28.

---

## Interactive live viewer (both modes)

- **Recorded**: `save_vis_video.py` writes a per-run mp4 into `OUT_DIR`, pulled small to the laptop
  (rule 4).
- **Live browser** (while a run is active): forward the viewer port over SSH and open the MJPEG
  dashboard in a browser:

```sh
ssh -L 8821:localhost:8821 <host>    # PushT   → http://localhost:8821
ssh -L 8801:localhost:8801 <host>    # MimicGen → http://localhost:8801
```

The dashboard tiles the current **observation | dream+tracks | dream | Jacobian field** as the
rollout runs.

---

## Runtime characterization (gfx1151, bf16, SDPA)

MimicGen closed-loop, `--sample-steps 10`, render 128, 2 views:

| chunk | latency | note |
|---|--:|---|
| cold start (infer #1) | **~342-350 s** | one-time: cotracker `torch.hub` fetch + model loads + kernel autotune |
| warm (infer #2+) | **~142 s / chunk** | 10 env steps per chunk (WAN 120-frame dream + cotracker 2-view tracks + Jacobian IDM) |
| anomalous stall | **~600-1460 s** | rare gfx1151 hipblaslt/mem hiccup - tolerated by the keepalive fix, not dropped |

Steady-state per-chunk latency is essentially the **same at `sample_steps=40` and `10`** (~142 s):
the WAN sampler's teacache / diffusion-forcing amortizes denoise steps, so the extra fidelity of 40
steps is nearly free per chunk - the win is quality, not a 4× cost. The verified `stack_d0` success
was **50 chunks / ~2.6 h wall** (1× cold start + 47 warm chunks + 2 anomalous stalls); a clean run
without stalls is **~1.8-2.0 h**.

The per-chunk cost is dominated by the WAN DiT denoise + Wan2.1 VAE conv3d decode (the same split
profiled for video-gen: DiT ≈ 67.7 %, VAE conv3d ≈ 32.2 % of wall - see `PORT_SUMMARY.md`), plus the
VGGT Jacobian IDM and the cotracker pass on the dreamed video. A `HIPBLAS_STATUS_NOT_SUPPORTED`
hipblaslt→unfused-cublas fallback (non-fatal) inflates several linear layers on gfx1151; the first
optimization lever is fewer denoise steps / shorter horizon.

**Eval-depth budgeting.** One chunk = 10 env steps, so an episode that runs the full horizon `H`
costs `⌈H/10⌉ × ~142 s` (successes terminate early; failures burn the full `H`). That makes
statistically-meaningful depth expensive on a single iGPU:

| depth | worst-case / episode | 9-task suite (1 warm server) |
|---|--:|--:|
| `H=100`, 1 demo | ~24 min | **~3.7-5 h** (sign-of-life; used above) |
| `H=200`, 3 demos | ~2.3 h | ~15-20 h (overnight) |
| `H=400`, 10 demos | ~16 h | multi-day - hogs a shared box (rule 0.5), not advised |

For headline success rates, run a deeper pass on **one or two representative tasks** as a monitored
job rather than the full suite. (The WebSocket drop that used to waste a whole task is now fixed - 
keepalive is disabled on both ends; see port fix #5.)

---

## Project-specific port fixes (rule 2.1)

All are hardware/version/upstream-drift patches, kept minimal and reproducible in the image build
(`scripts/patch_eval_imports.py`, applied as a cheap late Docker layer since VERA is an editable
install - iterating on them never re-triggers the dependency reinstall).

1. **NVlabs mimicgen (editable, for its object assets), not the PyPI stub.** The PyPI project named
   `mimicgen` is unrelated (no `mimicgen.utils`); VERA's runner needs
   `mimicgen.utils.robomimic_utils.create_env`, so the Dockerfile unpins `mimicgen==1.0.0` and
   installs `NVlabs/mimicgen` `--no-deps` (+ `gdown`). It installs **editable** (`-e /repos/mimicgen`)
   rather than the `git+` form on purpose: mimicgen's per-task object XMLs + meshes
   (`mimicgen/models/robosuite/assets/`, ~12 MB, in-tree) are **not declared as package_data**, so a
   `git+` build silently drops them and any task loading a mimicgen-specific object crashes - e.g.
   coffee → `FileNotFoundError: .../objects/coffee_pod.xml`. Editable keeps `XML_ASSETS_BASE_PATH`
   pointing into the clone where the assets live, which is what unlocked the coffee family.
2. **torchmetrics 1.4.x LPIPS import guard.** `NoTrainLpips` / `_valid_img` moved/renamed; guarded
   so the eval-only metrics never hard-fail the policy import (metrics are off the inference path).
3. **Tracker backend field-name bug.** `tracker_backend_from_cfg` read `cfg.tracker_backend`, but
   the WAN pipeline's `MotionTrackConfig` names the field `backend`, so `MotionTrackConfig(
   backend="cotracker")` was silently ignored and the motion tracker always fell back to
   **alltracker** (a vendored-only checkpoint that VERA's own `build_policy` warns gives
   wrong-direction flow on MimicGen). Patch honors `.backend`; `demo_mimicgen.sh` layers
   `tracker.backend: cotracker` onto a **copy** of the shipped `algo_config.yaml` (never edits the
   downloaded asset). cotracker is `torch.hub`-fetched at runtime - no vendored binary.
4. **cotracker visualization dtype+device.** `draw_pts_gpu` runs on `rgbs.device` and
   boolean-indexes the visibility mask (`trajs[mask, t]`). The cotracker wrapper passed float
   visibility on CPU while the video stayed on GPU → `IndexError` (float mask) then a cuda/cpu
   device mismatch. Patch thresholds visibility to a bool mask kept on the video's device and hands
   `draw_pts_gpu` the GPU tracks (a CPU float copy is used only for the color numpy).
5. **WebSocket keepalive disabled on both ends (gfx1151 chunk stalls).** Upstream already raised the
   ping to 60 s/600 s for the 14B planner, but at `sample_steps=40` a single closed-loop chunk
   occasionally stalls past 600 s on `gfx1151`, tripping the keepalive mid-denoise and killing the
   episode; the EGL→OSMesa retry then can't reconnect (server still finishing the stalled chunk →
   handshake `TimeoutError`). The server *does* finish the chunk, so `patch_eval_imports.py` sets
   `ping_interval=None` on the client (`websocket_policy_client.py`) and `ping_interval=None,
   ping_timeout=None` on the server (`websocket_policy_server.py`). A merely-slow forward pass no
   longer drops the link; a truly dead server still surfaces (localhost TCP close), and the client's
   finite `open_timeout` still bounds a bad *initial* connect. Off the inference math entirely; this
   is what let a full `stack_d0` episode complete through two >600 s stalls.
