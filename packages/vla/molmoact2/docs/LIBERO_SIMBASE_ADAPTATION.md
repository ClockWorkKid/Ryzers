# MolmoAct2 → shared `simulation/libero` base - adaptation design

> Status: **design + adapter skeleton on the spin-off branch `benchmark-molmoact2-libero`.**
> Not yet merged into `benchmark`. The dependency reconciliation and closed-loop parity
> MUST be validated on `strix-halo` (where MolmoAct2 + the `MolmoAct2-*-LIBERO` checkpoints
> are already validated) before this is brought into `benchmark`.

## Goal

MolmoAct2 predates the sim-base architecture: it runs LIBERO **entirely through the
allenai/lerobot eval stack** (`lerobot-eval --policy.type=molmoact2 --env.type=libero`,
`lerobot.scripts.lerobot_eval.rollout`), bundling its own LIBERO env, interactive servers,
and action-dump patch. This task reworks it to consume the shared **`simulation/libero`**
base through the model-agnostic `sim_libero.Policy` seam + `POLICY_FACTORY`, mirroring
`wam/fastwam`'s adapter - so every model runs closed-loop/interactive LIBERO on one
canonical harness.

## The seam (target contract)

`simulation/libero` ships a model-agnostic harness (`/opt/sim/sim_libero`, on `PYTHONPATH`)
that owns the env, rendering, streaming and the episode loop. A model plugs in by:

1. Implementing `sim_libero.policy.Policy`:
  - `predict_action_chunk(obs, instruction) -> np.ndarray [T, 7]` where the 7-D action is
     LIBERO OSC-delta `[dx, dy, dz, droll, dpitch, dyaw, gripper]`. `obs` is the **raw
     robosuite obs dict** returned by `scene.reset()` / `env.step()` (keys:
     `agentview_image`, `robot0_eye_in_hand_image`, `robot0_eef_pos`,
     `robot0_eef_quat`/`..._axis_angle`, `robot0_gripper_qpos`, ...).
  - optional `reset(instruction)` (clear the per-episode depth/spatial-plan cache),
     `replan_steps`, `num_steps_wait`, `name`.
2. Exposing `build_policy() -> Policy`, selected at runtime by
   `POLICY_FACTORY=molmoact2_libero_policy:build_policy`.

The harness's `sim_libero.rollout.run_episode` settles `num_steps_wait` no-ops, then loops:
`predict_action_chunk(obs, instruction)` → execute the first `replan_steps` actions →
re-predict. Interactive/RT servers call the same seam.

## The crux - two Python stacks that must be reconciled

| | shared `simulation/libero` base | MolmoAct2 LIBERO (current) |
|---|---|---|
| numpy | `1.26.4` (robosuite numba, no py3.12 numpy-2 wheel) | `2.x` |
| sim stack | `robosuite==1.4.0`, `mujoco==3.3.2`, `gym==0.25.2`, LIBERO @ HEAD | pulled by `lerobot[libero]` (its own pins) |
| transformers | none (model-agnostic; harness never imports it) | **`5.3`** (allenai/lerobot fork + `MolmoAct2-*-LIBERO`), isolated in `/opt/libero-venv` to avoid clobbering the base `/opt/venv` transformers `4.57` used by the DROID path |
| model deps | none | allenai `molmoact2` @ `cdf4b772`, allenai `lerobot` @ `molmoact2-hf-inference` |

`POLICY_FACTORY` loads the policy factory **into the harness process**, so the lerobot
MolmoAct2 *policy* and the `sim_libero` harness must co-exist in one interpreter. Key
observation that makes this tractable: **we only need lerobot's policy, not its env.** The
shared harness owns the env, so we can install `lerobot` (core) WITHOUT the `[libero]`
extra, avoiding the robosuite/LIBERO sim-stack clash. Only the policy path
(`make_policy` + `make_pre_post_processors` + the molmoact2 policy class + transformers 5.3)
needs to load.

### Open questions to settle on `strix-halo` (validation gates)

1. **Single-process import**: does `transformers==5.3` + allenai `lerobot` (core, no
   `[libero]`) + `molmoact2` co-exist with the base's `numpy==1.26.4` / `robosuite==1.4.0`
   in one env? Or does transformers 5.3 force `numpy>=2`? If it forces numpy 2, fall back
   to the **policy-server bridge** (below).
2. **Obs mapping**: the exact observation keys/shape/dtype the MolmoAct2 lerobot policy
   preprocessor expects (`observation.images.image`, `observation.images.wrist_image`,
   `observation.state` composition + dim, `task`) and the **image orientation** (the
   lerobot LIBERO env applies its own rotation; `sim_libero.get_libero_image` rotates 180°
   to match FastWAM training - confirm MolmoAct2's expected orientation).
3. **Action decode**: confirm the postprocessor returns LIBERO OSC-delta `[7]` directly
   (`norm_tag=libero` un-normalization) and how to pull a **full chunk** `[T,7]`
   (`predict_action_chunk` vs. draining the policy's receding-horizon action queue).
4. **Closed-loop parity**: success-rate on a small suite slice must match the current
   `lerobot-eval` path (the `demo_libero.sh` numbers) within noise.

### Fallback: policy-server bridge (if single-process is infeasible)

If (1) fails, keep MolmoAct2's isolated `/opt/libero-venv` and run the policy as a small
local server there; the `sim_libero`-side adapter (base env) becomes a thin client that
POSTs `(obs, instruction)` and receives `[T,7]`. This is the same split-Python pattern
noted for `simulation/simplerenv`. Heavier, but robust to the transformers/numpy split.

## Package changes (this branch)

- `adapters/molmoact2_libero_policy.py` - the `sim_libero.Policy` bridge (skeleton here;
  model-API specifics flagged `# VALIDATE-ON-BOX`).
- `config.yaml` - add the `sim_libero` knobs (`SUITE`, `TASK_ID`, `SEED`, `PORT`,
  `VIEW_RES`, `VIDEO_RES`, `RENDER_RES`, `MAX_STEPS`, `RT_HZ`, `REPLAN_STEPS`,
  `NUM_STEPS_WAIT`, `POLICY_FACTORY`), mirroring `wam/fastwam`.
- `demos/demo_closedloop_libero.sh`, `demos/demo_interactive_libero.sh`,
  `demos/demo_interactive_libero_rt.sh` - chain form
  (`ryzers build simulation/libero molmoact2`), set `POLICY_FACTORY`, run the base harness
  demos (`/ryzers/demos/demo_*` from the base).
- Dockerfile - a chain-aware branch: when built on the `simulation/libero` base
  (`/opt/LIBERO` present), install the lerobot **policy** deps (no `[libero]` extra) and put
  `adapters/` on `PYTHONPATH`. The existing DROID/interactive (non-LIBERO) stack stays.
- The bundled LIBERO harness (`scripts/interactive_server*.py`, `libero_action_plot.py`,
  `demos/demo_libero.sh`, `apply_action_patch.py`, the `/opt/libero-venv`) is **retired for
  the shared-base path** once parity is validated; keep until then for A/B comparison.

## Validation checklist (strix-halo)

- [ ] `ryzers build simulation/libero molmoact2` builds; single-process import guard passes.
- [ ] `POLICY_FACTORY=molmoact2_libero_policy:build_policy` loads the policy in the harness.
- [ ] Sanity: one interactive episode renders + executes actions (no shape/orientation errors).
- [ ] Closed-loop success rate on `libero_object` (few tasks) matches the `lerobot-eval` path.
- [ ] Interactive + RT servers drive MolmoAct2 through the base harness.
