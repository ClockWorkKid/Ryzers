# avdc - AVDC video-diffusion policy (ROCm, gfx1151)

AVDC (flow-diffusion) - *"Learning to Act from Actionless Videos through Dense Correspondences"* -
ported to AMD Strix Halo (gfx1151, ROCm). A text-conditioned 3D-UNet **video-diffusion policy**:
given the current frame + a task string it imagines a short future video, then turns that video
into actions via dense optical flow. This package is the general AVDC model layer and its native
demos (open-loop video prediction + closed-loop Meta-World). The iTHOR ObjectNav consumer lives in
`packages/wam/avdc-ithor`, which chains on the standalone `packages/simulation/ithor` sim base.

## What's in the image
- **Model**: 3D-UNet video diffusion (`flowdiffusion` + vendored `guided_diffusion` UNet, task-token
  cross-attention), OpenAI CLIP ViT-B/32 text encoder -> 512-d task tokens, GoalGaussianDiffusion
  (imagen-pytorch style, `pred_v`, DDIM). All attention is plain matmul/softmax - nothing
  ROCm-specific to patch in the forward math (rule 2.1).
- **Closed-loop sim**: vendored **Meta-World** (mujoco-py / MuJoCo 2.1) rendered headless via **EGL**
  on the iGPU, plus **UniMatch** (GMFlow) dense optical flow. Single self-contained image (rule 0.4).
- **Weights** (CLIP + AVDC snapshots + UniMatch) are fetched at runtime / build from upstream, never
  re-hosted (rule 8).

## Pipeline (closed-loop, from upstream benchmark_mw.py)
```
current frame + task string
  -> AVDC 3D-UNet imagines a short future clip
  -> UniMatch GMFlow dense optical flow between generated frames
  -> flow + depth/segmentation solved into an end-effector rigid transform
  -> robot actions; execute, re-plan on drift, until solved or the step cap
```
Open-loop (`demo_openloop`) stops after the first arrow: one frame + task -> predicted future video
(sim-agnostic, no simulator).

## Build + run
```sh
# Build on the ROCm base (packages resolve by leaf name):
ryzers build avdc --name avdc

# Fetch checkpoints (rule 8). WHICH = metaworld | metaworld-DA | ithor | all (default: MW + MW-DA):
WHICH=metaworld ryzers run --name avdc /ryzers/scripts/download_checkpoints.sh

# Open-loop video prediction (init frame + task -> generated future; rule-2.b layout):
IMAGE=/data/init.png TEXT="open the door" ryzers run --name avdc /ryzers/demos/demo_openloop.sh

# Closed-loop Meta-World rollout (headless EGL) -> executed GIF + metrics:
ENV_NAME=door-open-v2-goal-observable ryzers run --name avdc /ryzers/demos/demo_metaworld.sh

# Full Meta-World benchmark sweep (11 tasks x seeds x cameras, single process):
ryzers run --name avdc /ryzers/demos/demo_benchmark.sh
```

## Knobs (env)
- **Model / checkpoint**: `CKPT_DIR` (`/models/metaworld`), `MILESTONE` (24), `SAMPLE_STEPS` (DDIM
  steps), `WHICH` (checkpoint set to fetch).
- **Open-loop**: `IMAGE`, `TEXT`, `GUIDANCE`, `FLOW` (flow-UNet variant), `SEED`.
- **Closed-loop Meta-World**: `ENV_NAME` (V2 goal-observable task), `CAMERA` (corner|corner2|corner3),
  `MAX_REPLANS`, `SEED`.
- **Optimization** (`avdc_optim`, env-gated): `AVDC_AMP` (fp16), `AVDC_COMPILE` (torch.compile the
  3D-UNet), `AVDC_COMPILE_MODE`.
- **ROCm / MIOpen**: `MIOPEN_FIND_MODE` - see the caveat below.

## Layout
- `Dockerfile` - AVDC model deps on the ROCm base + the Meta-World (mujoco-py) + UniMatch layers.
- `avdc_optim.py` - env-gated fp16 autocast + `torch.compile` (shared with the iTHOR consumer).
- `test.py` - weight-free forward sign-of-life (the image `CMD`).
- `demos/` - `demo_openloop.*` (open-loop), `demo_metaworld.*` (closed-loop), `demo_benchmark.*`
  (MW sweep). See `demos/README.md`.
- `patches/rocm_port.py` - minimal ROCm/API source patches over the pinned upstream.
- `scripts/download_checkpoints.sh` - runtime weight fetch (rule 8).
- `docs/UPSTREAM_PIN.commit.txt` - pinned upstream commits + weight URLs.

## Validated on AMD Strix Halo (gfx1151, ROCm 7.2.2)
- **Forward smoke** (`test.py`): 3D-UNet produces the expected video-tensor shape on the ROCm stack.
- **Closed-loop Meta-World**: `door-open-v2-goal-observable`, seed 0, corner camera, 20 DDIM steps,
  fp16 -> **success=True**, episode return **374.9**, eplen 103, 0 replans; executed rollout GIF +
  metrics written. Headless EGL on the iGPU works end-to-end (video model -> UniMatch flow ->
  end-effector actions -> MuJoCo).

## Notes / caveats
- **MIOpen find-mode**: the Meta-World 3D-UNet (128x128 x 7 frames) has heavy 3D convs. The default
  `MIOPEN_FIND_MODE=NORMAL` runs an exhaustive kernel search that is *very* slow to warm on gfx1151
  (first denoise step can stall for many minutes). For quick runs use `MIOPEN_FIND_MODE=FAST`
  (immediate kernels; ~50 s door-open rollout). NORMAL's tuned kernels are cached in the mounted
  MIOpen dir, so if you want peak throughput pay the one-time tune once and reuse it.
- `torch.compile` (`AVDC_COMPILE=1`) adds a one-time inductor warmup; skip it (`AVDC_COMPILE=0`) for
  short smoke runs.
- `numpy` is pinned `1.26.4` image-wide (mujoco-py / gym 0.25 predate numpy 2); torch + transformers
  + flowdiffusion all run fine on it.
- Weights fetched at runtime/build from upstream, never re-hosted (rule 8); large videos stay on the
  remote (rule 3).
