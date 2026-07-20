# avdc - AVDC video-diffusion policy (ROCm, gfx1151)

AVDC (flow-diffusion) - *"Learning to Act from Actionless Videos through Dense Correspondences"* -
ported to AMD Strix Halo (gfx1151, ROCm). A text-conditioned 3D-UNet **video-diffusion policy**:
given the current frame + a task string it imagines a short future video, then turns that video
into actions via dense optical flow.

This is a **single consolidated package**: one image covers open-loop video prediction and *both*
closed-loop benchmarks (Meta-World and iTHOR ObjectNav). It is built **on top of the
`packages/simulation/ithor` sim base**, so the iTHOR half reuses that base's ai2thor + Vulkan
runtime + `sim_ithor` harness instead of duplicating it; the Meta-World MuJoCo sim is vendored into
this layer.

## What's in the image
- **Model**: 3D-UNet video diffusion (`flowdiffusion` + vendored `guided_diffusion` UNet, task-token
  cross-attention), OpenAI CLIP ViT-B/32 text encoder -> 512-d task tokens, GoalGaussianDiffusion
  (imagen-pytorch style, `pred_v`, DDIM), and **UniMatch** (GMFlow) dense optical flow.
- **iTHOR** closed-loop: inherited from the sim base (ai2thor CloudRendering / Vulkan + `sim_ithor`).
- **Meta-World** closed-loop: vendored **MuJoCo 2.1** (mujoco-py / gym) rendered headless via EGL.
- **Weights** (CLIP + AVDC snapshots) fetched at runtime; UniMatch + MuJoCo fetched at build. Nothing
  re-hosted (rule 8).

## Pipeline (closed-loop)
```
current frame + task string
  -> AVDC 3D-UNet imagines a short future clip
  -> UniMatch GMFlow dense optical flow between generated frames
  -> flow + depth solved into actions (end-effector transform for MW; discrete nav for iTHOR)
  -> execute, re-plan on drift, until solved / target visible / step cap
```
Open-loop (`demo_openloop`) stops after the first arrow: one frame + task -> predicted future video.

## Build + run
```sh
# Build ON the iTHOR sim base (packages resolve by leaf name; sim base first):
ryzers build ithor avdc --name avdc

# Fetch checkpoints (rule 8). WHICH = metaworld | metaworld-DA | ithor | all:
WHICH=all ryzers run --name avdc /ryzers/scripts/download_checkpoints.sh

# Open-loop video prediction (init frame + task -> generated future; rule-2.b layout):
IMAGE=/data/init.png TEXT="open the door" ryzers run --name avdc /ryzers/demos/demo_openloop.sh

# Closed-loop Meta-World rollout (headless EGL) -> executed GIF + metrics:
ENV_NAME=door-open-v2-goal-observable ryzers run --name avdc /ryzers/demos/demo_metaworld.sh

# Closed-loop iTHOR ObjectNav rollout (Vulkan) -> rollout video + two-column plan-vs-sim GIF:
SCENE=FloorPlan1 TARGET=Toaster ryzers run --name avdc /ryzers/demos/demo_ithor.sh

# Full benchmark sweeps (single process; compile warmup amortized):
ryzers run --name avdc /ryzers/demos/demo_metaworld_benchmark.sh      # Meta-World
TASKS=all N_SEEDS=20 ryzers run --name avdc /ryzers/demos/demo_ithor_benchmark.sh   # iTHOR
```

## Knobs (env)
- **Checkpoint**: `WHICH`, `CKPT_DIR`, `MILESTONE`, `SAMPLE_STEPS` (per-demo defaults: MW
  `/models/metaworld` m24, iTHOR `/models/ithor` m16).
- **Open-loop**: `IMAGE`, `TEXT`, `GUIDANCE`, `FLOW`, `SEED`.
- **Meta-World**: `ENV_NAME`, `CAMERA` (corner|corner2|corner3), `MAX_REPLANS`, `SEED`.
- **iTHOR**: `SCENE`, `TARGET`, `SEED`, `TASKS`, `N_SEEDS`, `MAX_EPLEN`, `RENDER_RESOLUTION`,
  `THOR_PLATFORM`, `VK_ICD_FILENAMES`, `POLICY_FACTORY` (default `avdc_ithor_policy:build_policy`).
- **Optimization** (`avdc_optim`, env-gated): `AVDC_AMP` (fp16), `AVDC_COMPILE`, `AVDC_COMPILE_MODE`.
- **ROCm / MIOpen**: `MIOPEN_FIND_MODE` - see the caveat below.

## Layout
- `Dockerfile` - `FROM` the iTHOR sim base; adds the AVDC model stack + UniMatch + Meta-World MuJoCo.
- `adapters/avdc_ithor_policy.py` - the `sim_ithor.Policy` adapter (video->flow->nav) for iTHOR.
- `avdc_optim.py` - env-gated fp16 autocast + `torch.compile` (shared by MW + iTHOR).
- `test.py` - AVDC model forward sign-of-life (installed as `/ryzers/model_smoke.py`, the image CMD;
  the base's Vulkan sim gate stays at `/ryzers/test.py`).
- `demos/` - `demo_openloop.*`, `demo_metaworld.*` + `demo_metaworld_benchmark.*`, `demo_ithor.*` +
  `demo_ithor_benchmark.sh` (drives the sim base's multi-task benchmark with the AVDC policy).
- `patches/rocm_port.py` - NVML no-op ROCm patch. `scripts/download_checkpoints.sh` - weight fetch.
- `docs/UPSTREAM_PIN.commit.txt` - pinned upstream commits + weight URLs.

## Validated on AMD Strix Halo (gfx1151, ROCm 7.2.2)
- **Model forward smoke** (`model_smoke.py`): 3D-UNet + DDIM sampling produce the expected
  video-tensor shape on the ROCm stack.
- **Closed-loop Meta-World**: `door-open-v2-goal-observable`, seed 0, corner, 20 DDIM steps, fp16 ->
  **success=True**, episode return **374.9**, eplen 103, 0 replans (headless EGL end-to-end).
- **Closed-loop iTHOR ObjectNav**: 12 tasks x 5 seeds = 60 episodes, fp16 + `torch.compile`,
  `MAX_EPLEN=50` -> mean success **0.233** (per-task 0.0-0.4), in line with AVDC's published iTHOR
  numbers. `avdc_optim` gives ~2x faster DDIM sampling (~18-22 it/s vs ~10 it/s), quality-neutral.

## Notes / caveats
- **MIOpen find-mode**: the Meta-World 3D-UNet (128x128 x 7 frames) has heavy 3D convs; the default
  `MIOPEN_FIND_MODE=NORMAL` runs an exhaustive search that is *very* slow to warm on gfx1151 (first
  denoise step can stall for many minutes). Use `MIOPEN_FIND_MODE=FAST` for quick runs (immediate
  kernels; ~50 s door-open rollout); NORMAL's tuned kernels are cached in the mounted MIOpen dir.
- `torch.compile` (`AVDC_COMPILE=1`) adds a one-time inductor warmup; skip it (`AVDC_COMPILE=0`) for
  short smoke runs.
- `numpy` is pinned `1.26.4` image-wide (inherited from the sim base; mujoco-py / gym 0.25 predate
  numpy 2). Weights fetched at runtime/build from upstream, never re-hosted (rule 8); large videos
  stay on the remote (rule 3).
