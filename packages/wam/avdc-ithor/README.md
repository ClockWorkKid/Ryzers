# avdc-ithor - AVDC as the iTHOR ObjectNav model layer (ROCm, gfx1151)

AVDC (flow-diffusion video-diffusion policy) driving AI2-THOR (iTHOR) ObjectNav, built ON TOP of
the model-agnostic `packages/simulation/ithor` sim base. The sim base owns ai2thor + the Unity
CloudRendering (Vulkan) runtime + the `sim_ithor` harness; this layer adds the AVDC model half and
a thin `sim_ithor.Policy` adapter. Mirrors the `packages/wam/vera` on `packages/simulation/mimicgen`
pattern.

## Pipeline (per plan step, from upstream benchmark_thor.py)
current frame + target
  -> AVDC 3D-UNet imagines a short future video (`pred_video_thor`, milestone-16)
  -> UniMatch GMFlow dense optical flow between generated frames (`pred_flow_frame`)
  -> flow + sim depth -> rigid transforms (`get_transforms_nav`)
  -> discrete nav actions `{MoveAhead, RotateLeft, RotateRight, Done}` (`transforms2actions`)
Execute the actions, re-plan, until the target is visible (success) or the step cap.

Same 3D-UNet as the Meta-World AVDC image, so `avdc_optim` (fp16 + torch.compile) applies directly.

## Build + run
```sh
# Build the model layer ON the sim base (packages resolve by leaf name; sim base first):
ryzers build ithor avdc-ithor --name avdc-ithor

# Fetch the iTHOR checkpoint (rule 8, not re-hosted):
ryzers run --name avdc-ithor /ryzers/scripts/download_checkpoints.sh

# Single ObjectNav rollout (+ rule-2.b two-column plan-vs-sim GIF):
SCENE=FloorPlan1 TARGET=Toaster ryzers run --name avdc-ithor /ryzers/demos/demo_ithor.sh

# Full benchmark (4 scenes x 3 targets x N seeds, single process):
TASKS=all N_SEEDS=20 ryzers run --name avdc-ithor /ryzers/demos/demo_benchmark.sh
```

## Knobs (env)
`SCENE`, `TARGET`, `SEED`, `TASKS`, `N_SEEDS`, `MAX_EPLEN`, `RENDER_RESOLUTION`, `SAMPLE_STEPS`
(DDIM steps), `CKPT_DIR` (/models/ithor), `MILESTONE` (16), `AVDC_AMP` (fp16), `AVDC_COMPILE` (1),
`THOR_PLATFORM`, `VK_ICD_FILENAMES`, `POLICY_FACTORY` (default `avdc_ithor_policy:build_policy`).

## Layout
- `Dockerfile` - FROM the sim base; adds flowdiffusion + UniMatch + CLIP + the adapter.
- `adapters/avdc_ithor_policy.py` - the `sim_ithor.Policy` adapter (video->flow->nav).
- `avdc_optim.py` - env-gated fp16 + torch.compile (shared with the MW AVDC image).
- `patches/rocm_port.py` - NVML no-op ROCm patch.
- `demos/demo_ithor.*` - single rollout + two-column viz. `demos/demo_benchmark.sh` - full sweep.
- `scripts/download_checkpoints.sh` - iTHOR `model-16.pt`.

## Validated on AMD Strix Halo (gfx1151, ROCm 7.2.2, Mesa 25.2 RADV)
- Headless Vulkan render gate: PASS (*Radeon 8060S Graphics (RADV GFX1151)*, driver radv).
- Benchmark: 12 tasks x 5 seeds = 60 episodes, fp16 + torch.compile, `MAX_EPLEN=50`.
  Mean ObjectNav success **0.233** (per-task 0.0-0.4), wall ~50 min single-process. In line with
  AVDC's published iTHOR performance.
- Optimization (`avdc_optim`): fp16 autocast + `torch.compile` on the 3D-UNet -> DDIM sampling
  ~2x faster (~18-22 it/s vs ~10 it/s; 100 steps ~4 s vs ~10 s). Quality-neutral: on shared
  task/seeds the optimized rollouts match the fp32 baseline episode lengths exactly. End-to-end
  episode wall is dominated by Unity rendering + optical flow, so the overall speedup is smaller
  than the sampling speedup. One-time `torch.compile` (~108 s) is amortized across the run.

## Notes
- Weights fetched at runtime (rule 8); large videos + the Unity build stay on the remote (rule 3).
- numpy pinned 1.26.4 (inherited from the sim base / AVDC stack).
