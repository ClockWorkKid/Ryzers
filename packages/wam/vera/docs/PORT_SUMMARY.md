# VERA Ryzer - Port Summary (progress log)

Upstream: github.com/sizhe-li/VERA @ `fe8561a` · bases: Wan2.1-T2V-1.3B / Wan2.1-I2V-14B-480P /
VGGT-1B · target: AMD Strix Halo (Ryzen AI Max+ 395, `gfx1151`), ROCm 7.2.2 · branch `wam-vera`
off `benchmark`.

## Phase 1 - image build + full package import smoke ✅ (Gate 1)
- Direct PyTorch → ROCm port: base image ROCm torch preserved; the upstream `torch==2.6.0` CUDA pin
  (ABI-tied to flash-attn wheels) is stripped via `scripts/strip_cuda_torch.py`, everything else
  installed under `PIP_CONSTRAINT`. flash-attn is not installed → the WAN planner falls back to
  SDPA/AOTriton on gfx1151. `numpy` is intentionally left unpinned (the VGGT IDM backbone requires
  `numpy<2`; ROCm torch 2.10 round-trips cleanly on 1.26.4).
- Result: `vera:latest` builds; `test.py` PASS on `Radeon 8060S` (torch 2.10.0+rocm7.2.2,
  transformers 5.13, diffusers 0.39); full `vera{,.policy,.idm,.server,.video_model,.controller}` +
  vggt import.

## Phase 2 - load a released checkpoint ✅ (Gate 2)
- Weights fetched at runtime (rule 8): `scripts/download_checkpoints.sh wave1` (MimicGen 1.3B +
  PushT) / `droid` (14B). Frozen bases auto-pull on first run.
- DROID WAN-14B planner (`WanPipeline.from_config`, bf16) loads in **54 s**, **46.9 GB VRAM** (of
  96 GB); context 29 frames → 24 generated/call.

## Phase 3 - video generation ✅ (Gate 3)
- `examples/droid_generation.ipynb` ported to headless `scripts/videogen_droid.py` +
  `demos/demo_videogen.sh`; one scene-2 prompt generated a 53-frame clip (29 ctx + 24 dream) that
  follows the prompt. **Wan2.1 VAE conv3d bf16 decode did NOT hang** on gfx1151.
- `scripts/videogen_image2video.py` + `demos/demo_img2vid.sh` (held input left / generated rollout
  right, rule 2.b) and `scripts/videogen_multiprompt.py` (fixed context, N language commands, fixed
  seed) added; artifacts mirrored small to the laptop.
- **Per-component profile** (`scripts/profile_videogen.py`, CUDA-event forward hooks + FlopCounter):
  one 29→24 generation, 40 denoise steps - **WanModel DiT 889 s (67.7% wall, 99.6% FLOPs)**,
  **WanVAE encode+decode ~423 s (32.2% wall, conv3d memory/kernel-bound)**, T5/CLIP trivial. Wall
  ~1313 s/gen, peak 49.2 GB. Levers: fewer denoise steps / shorter horizon, DiT attention kernel,
  faster conv3d decode.

## Phase 4 - closed-loop control + interactive viewer ✅ (see `CLOSEDLOOP.md`)
- Rebuilt to `.[idm,video,eval]` + EGL/OSMesa GL libs for headless MuJoCo. VERA source patches are
  applied as a **cheap late Docker layer** (editable install), so patch iterations never re-trigger
  the dependency reinstall.
- **PushT closed-loop ✅** - `demos/demo_pusht.sh` + headless `scripts/closedloop_pusht.py` (ported
  from `examples/pusht_dfot_stack.ipynb`): server (DFoT planner + Jacobian IDM, no MuJoCo) + client
  + `save_vis_video`. 3 rollouts from replay state 3664 (horizon 200, seed 42): **100% success,
  100% mean max-reward, 82% mean coverage**.
- **MimicGen closed-loop ✅ (full task range)** - `demos/demo_mimicgen.sh` (parametrized by `TASK`)
  and `demos/demo_mimicgen_suite.sh` (all 9 core tasks against **one warm server**) drive
  robosuite/MuJoCo via the upstream `run_mimicgen_eval` controller + WAN-1.3B planner + MimicGen
  Jacobian IDM, rendering headless under **EGL** with an **OSMesa** retry. All four task families
  run - **coffee / square / stack / stack_three**; the nine `*.hdf5` are fetched from
  `amandlek/mimicgen_datasets` (rule 8) - the download script now lives in the sim base package
  (`packages/simulation/mimicgen/scripts/download_mimicgen_datasets.sh`; see Phase 5).
- **Sign-of-life suite** (`NUM_DEMOS=1 ROLLOUT_HORIZON=100`, one warm server, ~5.2 h): **8/9 tasks
  `rc=0`** - load, step closed-loop, render end-to-end, each with viewer + per-view clips. Only
  `square_d2` failed, on an intermittent >600 s keepalive stall (both backends); `square_d0/d1`
  recovered via the OSMesa retry. Success rate 0/1 across the board is expected at this depth (100
  steps ≪ task horizons) - this validates the *pipeline* per task, not policy skill. Latency: cold
  start ~342-350 s (incl. one-time cotracker `torch.hub` fetch + loads), warm **~142 s/chunk** (10
  env steps); real success rates need many demos × full horizon (days on one gfx1151 - see
  `CLOSEDLOOP.md` eval-depth budgeting).
- **Port fixes (rule 2.1)**, all reproducible in the image build (`scripts/patch_eval_imports.py`):
  1. **NVlabs mimicgen (editable, for object assets), not the PyPI stub** - the PyPI project named
     `mimicgen` is unrelated (no `mimicgen.utils`); VERA's runner needs
     `mimicgen.utils.robomimic_utils.create_env`, so the Dockerfile unpins it and installs
     `NVlabs/mimicgen` `--no-deps` (+ `gdown`), **editable** (`-e`) so its per-task object XMLs+meshes
     (`models/robosuite/assets/`, not declared as package_data) aren't dropped - a `git+` build lost
     them and coffee crashed on `FileNotFoundError: objects/coffee_pod.xml`. Editable unlocked coffee.
  2. **torchmetrics 1.4.x LPIPS import guard** - `NoTrainLpips` / `_valid_img` moved/renamed;
     guarded so the eval-only metrics never hard-fail the policy import (off the inference path).
  3. **Tracker backend field-name bug** - `tracker_backend_from_cfg` read `cfg.tracker_backend`,
     but the WAN pipeline's `MotionTrackConfig` names the field `backend`, so `backend="cotracker"`
     was silently ignored and the tracker always fell back to the vendored-only **alltracker**
     (which VERA's own `build_policy` warns gives wrong-direction flow on MimicGen). Patch honors
     `.backend`; the demo layers `tracker.backend: cotracker` onto a *copy* of the shipped
     `algo_config.yaml`. cotracker is `torch.hub`-fetched at runtime - no vendored binary.
  4. **cotracker visualization dtype+device** - `draw_pts_gpu` runs on `rgbs.device` and
     boolean-indexes the visibility mask; the cotracker wrapper passed float visibility on CPU while
     the video stayed on GPU (→ `IndexError` then a cuda/cpu mismatch). Patch thresholds visibility
     to a bool mask on the video's device and hands `draw_pts_gpu` the GPU tracks.
- A non-fatal `libGLU.so.0` EGL warning fires at MimicGen env-close (cosmetic; offscreen render + all
  video dumps succeed) - add `libglu1-mesa` at the next rebuild to silence it.

## Phase 5 - model/simulator package split ✅ (validated)
- Split the monolith into a **model-agnostic simulator base** and a **model layer**, along the
  websocket obs→action-chunk boundary that already separated them:
 - **`packages/simulation/mimicgen`** - installs robosuite/robomimic/NVlabs-mimicgen/MuJoCo + the
    sim/eval half of VERA (`vera[eval]`), and ships the `sim_mimicgen` harness: a `Policy` seam
    (`BasePolicy` + built-in `RandomPolicy` for a no-model sanity, and `RemoteWebsocketPolicy` for a
    running policy server) wrapping VERA's upstream `MimicgenRunner` unchanged (rule 2.1). Owns the
    dataset download script and its own `/sim_outputs` + `/sim_data` mounts.
 - **`packages/wam/vera`** - now builds `FROM` the sim base (`ryzers build … mimicgen vera`, which
    threads `BASE_IMAGE` ryzer_env→mimicgen→vera), adding only `.[idm,video]` + the policy server +
    the gfx1151 speedup patches + a thin adapter (`adapters/vera_mimicgen_policy.py`,
    `POLICY_FACTORY=vera_mimicgen_policy:build_policy`) that plugs VERA into the sim seam.
 - New seam demo `demos/demo_closedloop_mimicgen.sh`: starts the VERA server, then drives the sim
    base harness against it over the websocket - exercises the packaged model/sim boundary end to end.
- **Validation (Strix Halo gfx1151, ROCm 7.2.2):**
 - Sim base built and validated **independently** (rule 2): `test.py` import smoke + `RandomPolicy`
    sanity rollout on `stack_d0` → per-view MP4s, no model.
 - `vera` chain built; import smoke + seam-import (`sim_mimicgen` + adapter) pass.
 - **MimicGen closed-loop via the seam** (`stack_d0`, WAN-1.3B planner + Jacobian IDM served over
    ws, cotracker backend): full rollout, `~7.9 s`/replan (10 env steps), vis + per-view MP4s.
 - **PushT closed-loop** (DFoT planner + IDM): **SR 100%** (state 3664, `max_reward=0.9985`,
    coverage 0.82, `success=1`), vis MP4.
- **Fix (rule 2.1):** `scripts/closedloop_pusht.py` read knobs via `int(os.environ.get(k, default))`,
  but the ryzers run script threads them as `-e VAR=${VAR:-}` → a present-but-**empty** string, which
  `.get` returns over the default (`int('')` → `ValueError`). Added an empty-tolerant `_env()` helper
  so a bare `ryzers run … demo_pusht.sh` works without exporting every knob.

## Next
- Dev full manual repro test → push `wam-vera` → `benchmark` (rule 0.3 approval).
- Deeper eval for headline success rates: one/two representative tasks at `H=400` × many demos as a
  monitored job (full-suite depth is a multi-day single-GPU hog - rule 0.5). Before any long run,
  harden the transport: the client WS keepalive is already 60/600 s, but a rare >600 s render/infer
  stall still drops a connection (cost `square_d2` above) - raise/disable client `ping_timeout`,
  prefer `MUJOCO_GL=osmesa` to skip the flaky EGL attempt, and add `libglu1-mesa`.
- Deferred embodiments: **Allegro / iiwa** closed-loop need upstream deps not yet packaged (Drake,
  the CRM toolkit, `neural_jacobian` modules) - separate porting effort.
- Post-milestone (efficiency): the per-chunk cost is planner-bound (WAN DiT denoise + Wan2.1 VAE
  conv3d) - sweep `--sample-steps` / horizon, attention kernel, and a bf16/decode pass toward
  faster closed-loop control on Strix Halo.
