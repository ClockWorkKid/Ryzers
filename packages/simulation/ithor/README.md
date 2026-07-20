# simulation/ithor - AI2-THOR (iTHOR) ObjectNav simulator base (ROCm, gfx1151)

Model-agnostic AI2-THOR simulator base for AMD Strix Halo, rendering headless via **Vulkan**
(ai2thor `platform=CloudRendering`, no X server). Ships only the simulator + a small harness
(`sim_ithor`): a `ThorEnv` ObjectNav wrapper, a model-agnostic `Policy` seam with a `POLICY_FACTORY`
selector, a generic rollout / sanity runner, and a built-in `ScriptedPolicy` (no model). Layer a
video/VLA model on top with `ryzers build ithor <model>` (see the AVDC consumer,
`packages/wam/avdc-ithor`). Validated headless on AMD Strix Halo: *Radeon 8060S Graphics (RADV
GFX1151)*.

Mirrors the `packages/simulation/{libero,mimicgen}` -> `packages/wam/vera` pattern.

## Why its own package
The simulator (Unity + Vulkan runtime + the ObjectNav env) is reusable across models and is heavy
to stand up (headless Unity on RADV). Keeping it separate lets any model plug in via the policy
seam without re-porting the sim.

## The task: iTHOR ObjectNav
4 scenes x 3 target objects (12 tasks). The agent starts at a random reachable pose and navigates
with discrete moves `{MoveAhead, RotateRight, RotateLeft, Done}` (rotateStepDegrees=45) until the
target object is `visible` (success) or the step cap (`max_eplen=50`) is hit. Depth is provided by
the sim (`renderDepthImage=True`). Task set + env are from upstream AVDC_experiments
`benchmark_thor.py` (rule 2.1).

## Headless rendering on AMD (the crux)
CloudRendering renders through Vulkan. On AMD that is **Mesa RADV** (`libvulkan1` loader +
`mesa-vulkan-drivers` ICD). On the ROCm 7.2.2 / Ubuntu 24.04 base (Mesa 25.2) the ICD is
`radeon_icd.json` (older Mesa used the arch-suffixed `radeon_icd.x86_64.json`; the Dockerfile sets
`VK_ICD_FILENAMES` to the former and symlinks the latter). gfx1151 (RDNA3.5) needs a recent Mesa, so
on Ubuntu 22.04 the image pulls the kisak-mesa PPA (skipped on 24.04+). The ai2thor Unity build
(~540 MB) is fetched from the allenai public bucket (rule 8) by the ai2thor `Controller` on first
run and cached in the mounted `~/.ai2thor` volume (not baked into the image, which the mount would
shadow).

Fallback ladder if CloudRendering does not come up: RADV -> AMDVLK ICD (`VK_ICD_FILENAMES=`) ->
newer ai2thor Unity build (`AI2THOR_VERSION` build-arg; client API unchanged) -> Xvfb + GLX.

## Quickstart
```sh
# Build + gate (headless Vulkan render spike):
ryzers build ithor --name sim-ithor
ryzers run --name sim-ithor

# Scripted ObjectNav rollout (no model), writes a video:
SCENE=FloorPlan1 TARGET=Toaster ryzers run --name sim-ithor /ryzers/demos/demo_sim_sanity.sh
```

## Plug in your own model
Ship a `build_policy() -> sim_ithor.Policy` factory implementing `plan(obs) -> list[str]` (see
`examples/template_policy.py`) and select it at runtime:
```sh
POLICY_FACTORY=your_module:build_policy \
  SCENE=FloorPlan1 TARGET=Toaster ryzers run --name your-ithor /ryzers/demos/demo_sim_sanity.sh
```

## Layout
- `Dockerfile` - ai2thor + Vulkan/RADV runtime + `sim_ithor` harness (Unity build fetched at first
  run into the mounted cache, not baked).
- `config.yaml` - env knobs (`SCENE/TARGET/N_SEEDS/...`, `THOR_PLATFORM`, `VK_ICD_FILENAMES`,
  `POLICY_FACTORY`) + `/sim_outputs`, `/sim_data`, `~/.ai2thor` mounts.
- `test.py` - headless Vulkan render gate.
- `lib/sim_ithor/` - `env.py` (ThorEnv CloudRendering), `tasks.py` (scene2targets, get_cmat),
  `policy.py` (Policy seam + loader), `rollout.py`, `scripted_policy.py`, `sanity.py`.
- `demos/` - `demo_sim_sanity.*`, `demo_benchmark.*`.
- `examples/template_policy.py` - copy-me model adapter.
- `scripts/prefetch_build.sh` - optional runtime warmer to pre-fetch the ai2thor Unity build into
  the mounted cache (rule 8; the `Controller` also fetches it automatically on first run).
- `docs/UPSTREAM_PIN.commit.txt`.

## Notes
- ROCm 7.2.2 base (torch 2.10) so a torch model layer chains cleanly; numpy pinned 1.26.4.
- Large videos + the Unity build stay on the remote (rule 3); only small artifacts to laptop
  `artifacts/` (rule 4).
