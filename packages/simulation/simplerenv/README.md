# simulation/simplerenv — SimplerEnv real-to-sim base (SAPIEN/Vulkan, ROCm/gfx1151)

Model-agnostic [SimplerEnv](https://github.com/simpler-env/SimplerEnv) (CoRL 2024) simulator
base for the ryzers chain, the **third simulation benchmark from the VLA-JEPA paper** (after
LIBERO and LIBERO-Plus). SimplerEnv evaluates real-world manipulation policies in simulation
on two embodiments — **Google Robot** and **WidowX + Bridge** — through a uniform Gym API.

> Status: **scaffold in place; base-image install path validated empirically.** First build
> attempt (below) established that this py3.12 base forces the SAPIEN 3 / ManiSkill3 route.
> The Dockerfile + `simplerenv_env.py` are being re-pointed to the SimplerEnv `maniskill3`
> branch (CPU physics backend) before the render smoke on gfx1151. Full harness
> (render/rollout/interactive) + VLA-JEPA adapter + closed-loop land after the render smoke.

## Backend decision (empirically forced): SAPIEN 3 / ManiSkill3 (CPU physics)

SimplerEnv has two backends and the choice is constrained by the base image:

- **ManiSkill2 + SAPIEN 2** would be the paper-faithful visual-matching path, BUT the first
  build proved **`sapien==2.2.x` ships no py3.12 wheel** (`pip` only offers `3.0.0.dev*`,
  `3.0.0b*`, `3.0.x`). The ryzers base is `rocm/pytorch` **py3.12**, so SAPIEN 2 is off the
  table without a separate py3.10 sub-env (which would split the model/sim Python and force
  a policy-server bridge — heavy).
- **ManiSkill3 + SAPIEN 3.0.0b1** *does* have py3.12 wheels (the exact version
  `simulation/robotwin` already runs on gfx1151), so we use it, with the **CPU physics
  backend** (`sim_backend="physx_cpu"`): ManiSkill3's GPU speedup relies on **PhysX GPU,
  which is CUDA/NVIDIA-only** and would not run on AMD anyway. SAPIEN *rendering* stays
  Vulkan (AMD-friendly, Mesa RADV + `/dev/dri`, proven by robotwin).
- Trade-off: ManiSkill3 real-to-sim envs may differ slightly from the ManiSkill2 numbers in
  the paper; we treat "close to paper" the same way as LIBERO-Plus and document the backend.

## Layout (mirrors simulation/robotwin + libero-plus)

```
simulation/simplerenv/
├── Dockerfile              # ARG BASE_IMAGE; Vulkan apt stack; sapien2 + ManiSkill2_real2sim + SimplerEnv; import smoke
├── config.yaml             # gpu_support, x11 off, HSA override, Vulkan ICD, /sim_outputs + /sim_models mounts, knobs
├── test.py                 # import sign-of-life (no GPU render)
├── README.md
├── demos/
│   └── demo_sim_sanity.sh  # headless RandomPolicy rollout -> MP4 (render validation)
├── scripts/
│   └── setup_simplerenv.sh # runtime real2sim asset fetch (not baked, compliance)
└── lib/sim_simplerenv/     # -> /opt/sim
    ├── __init__.py         # exports Policy, load_policy
    ├── envutil.py          # empty-string-safe env readers
    ├── policy.py           # Policy ABC + load_policy() + POLICY_FACTORY seam (7-D action)
    ├── random_policy.py    # default no-model policy
    ├── simplerenv_env.py   # Gym-API loader: build/reset/step/image/instruction/horizon, both embodiments
    └── sanity.py           # headless render/sim sanity (make-or-break gfx1151 validation)
```

Still to add after the render smoke passes: `render.py`, `rollout.py`,
`interactive_server.py`, `interactive_server_rt.py` (mirror robotwin/libero-plus).

## Policy seam

A policy chains on top and implements `sim_simplerenv.Policy.predict_action_chunk(obs,
instruction) -> [T, 7]`, where the 7-D action is `[dx, dy, dz, drot_axangle(3), gripper]`
(same shape as LIBERO's OSC-delta, so VLA-JEPA reuses its LIBERO action handling). Select it
with `POLICY_FACTORY=module:function`; default is the built-in `RandomPolicy`.

## Build / run

```sh
ryzers build simulation/simplerenv                       # base image
TASK=google_robot_pick_coke_can ryzers run /ryzers/demos/demo_sim_sanity.sh
# chained with a policy (planned):
ryzers build simulation/simplerenv vlajepa
ryzers run /ryzers/demos/demo_closedloop_simplerenv.sh   # (adapter TBD)
```

## VLA-JEPA eval target (paper Table 2)

Google Robot (Fractal ckpt): pick_coke_can, move_near, open/close drawer,
long_horizon_apple_in_drawer. WidowX (BridgeV2 ckpt): spoon_on_towel, carrot_on_plate,
stack_cube, put_eggplant_in_basket. Success is computed post-hoc from rollout videos
(`examples/SimplerEnv/eval_files/.../calc_metrics_evaluation_videos.py` upstream).

## Next steps

1. **DONE (empirical):** first build showed `sapien==2.2.x` has no py3.12 wheel → pivot to
   SAPIEN 3 / ManiSkill3 branch (above).
2. Re-point the Dockerfile to `pip install "sapien==3.0.0b1" mani_skill` + clone SimplerEnv
   `maniskill3` branch; adjust the import smoke (`import mani_skill`, gymnasium env ids).
3. Re-point `simplerenv_env.py` to the ManiSkill3 env API (gymnasium `make(..., obs_mode,
   sim_backend="physx_cpu")`, image extraction, task ids) — the ms2 `simpler_env.make` /
   `get_image_from_maniskill2_obs_dict` helpers change on ms3.
4. **Render smoke on gfx1151** (needs GPU): `sim_simplerenv.sanity` → MP4 (make-or-break for
   SAPIEN Vulkan offscreen on this box).
5. After render passes: add `render.py`/`rollout.py`/`interactive_server*.py`, the vlajepa
   `simplerenv` adapter (Fractal/Bridge ckpts, 7-D action reuse), `closedloop_simplerenv.py`,
   demos, then benchmark vs paper Table 2 and pin `SIMPLERENV_COMMIT`.
