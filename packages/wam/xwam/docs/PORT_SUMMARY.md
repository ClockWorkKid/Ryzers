# X-WAM Ryzer — Port Summary (progress log)

Upstream: github.com/sharinka0715/X-WAM @ `72cfb86` · base: Wan2.2-TI2V-5B ·
target: AMD Strix Halo (`gfx1151`), ROCm 7.2.2 · branch `wam-XWAM` off `benchmark`.

## Phase 1 — image build + ROCm env sign-of-life ✅ (Gate 1)
- Direct PyTorch → ROCm port: base image ROCm torch preserved (X-WAM's `requirements.txt`
  comments out torch/flash-attn; installed under a torch constraint). flash-attn has no
  gfx1151 wheel → `modules/attention.py` falls back to torch SDPA (validated on-device).
- Build-time guard checks ROCm torch + heavy deps only (no GPU at build time); the full
  X-WAM import graph + attention are validated at run time by `test.py`.
- Result: `xwam:latest` builds; `test.py` PASS on `Radeon 8060S Graphics`
  (torch 2.10.0+rocm7.2.2, hip 7.2.5, transformers 4.51.3, diffusers 0.39.0).

## Phase 2 — full-model forward with released weights ✅ (Gate 2)
- Weights fetched at runtime (rule 8): `Wan-AI/Wan2.2-TI2V-5B` base (~32 GB: UMT5-XXL enc +
  Wan2.2 VAE + DiT shards + tokenizer) + `sharinka0715/X-WAM-checkpoints/robotwin_sft`
  (~37 GB DeepSpeed `mp_rank_00_model_states.pt` + `config.yaml`).
- `scripts/model_smoke.py` builds `XWAMRunner(robotwin_sft)`, loads the SFT state dict, and runs
  one `generate(..., early_stop=True)` (the fast async-denoising action path) on a synthetic
  3-view observation. The "newly initialized extra_blocks / TRAIN this model" note from
  `from_pretrained` is expected — those depth/action/proprio weights are then overwritten by
  the SFT `load_state_dict`.
- Result (denoise: video=50, action=10, decoupled): model 13.06B params, load 48.9s,
  action chunk (32, 14) finite; first-call 135s (one-time ROCm warmup), steady-state ~10.0s,
  peak 29.4 GB VRAM. HxW=256x320, views=3, proprio_dim=16, action_dim=14, action_num=4.

## Phase 3 — open-loop eval on X-WAM-RoboTwin ✅ (Gate 3)
- `scripts/download_datasets.sh` fetches a small subset (5 episodes,
  `chunk-0000/episode_000000[0-4]`: RGB + depth for head/left/right cameras + action JSON) into
  the mounted HF cache (rule 8).
- `scripts/openloop_replay.py` reuses the **upstream `data.robot_dataset.RobotDataset`** loader
  (built 561 valid clips from the 5 episodes), instantiates `XWAMRunner(robotwin_sft)` from the
  Wan base, loads the SFT state dict, then feeds each clip's first observation through
  `generate(..., early_stop=True, run_depth=False)` and compares the predicted vs GT action
  chunk in the same quantile-normalized space. Reports per-dim normalized MAE + a GT(solid)-vs-
  pred(dashed) overlay on shared axes (rule 2.a). `demos/demo_openloop.sh` sequences it.
- Result (denoise: video=50, action=10, decoupled): model load 64.2s / 13.06B params;
  **mean normalized MAE 0.0417** over 5 clips; first clip 135.8s (warmup), then ~10.2s/clip.
  Idle right arm near-perfect (R_* ≈ 0.004–0.012); acting left arm higher (L_z=0.131,
  L_ax=0.126) — the action head tracks the motion arc with a small phase/amplitude lag.
  Plots → `/outputs` → laptop `artifacts/xwam/`.

## Phase 4a — closed-loop RoboTwin 2.0 rollout ✅ (Gate 4a)
- Direct in-process `Policy` adapter (`experiments/robotwin/xwam_policy/deploy_policy.py`): a
  `DirectXWAM` wrapper builds `XWAMRunner(robotwin_sft)`, loads the SFT state dict once, and runs
  `generate(..., early_stop=True)` per replan. This **bypasses the upstream zmq broker→server→
  client** path — inference is inlined in the RoboTwin process (single GPU, headless SAPIEN).
- `XwamRoboTwinPolicy` implements the shared seam (`get_model`/`eval`/`reset_model`): stacks the
  3 views → `[V,C,256,320]`, quantile-normalizes proprio, denormalizes the `(32,14)` chunk, and
  drives the sim with `take_action(..., action_type="ee")` (14-D per-arm EE deltas integrated in
  RoboTwin coords, matching upstream X-WAM). Action queue replans every `replan=16` steps.
- ROCm-specific fixes isolated to the X-WAM layer: pin `numpy==1.26.4` (mplib 0.2.1 native ABI —
  numpy 2.x from the X-WAM layer segfaulted at scene setup); runtime shim on `MplibPlanner`
  (`plan_path`/`plan_pose`/`plan_screw`) to accept/ignore RoboTwin's `constraint_pose` kwarg and
  catch TOPP parameterization failures (return `Fail` instead of crashing); `XWAM_PLAN_MODE=screw`
  dispatches to fast interpolative planning (full RRT was ~30–40 min/episode).
- Result: `beat_block_hammer`, `demo_clean`, 1 episode → **Success rate 1/1 = 100%**. Rollout
  video + `_result_clean.txt` → laptop `artifacts/xwam/p4a_beat_block_hammer_success.mp4`
  (+ 4-frame montage `p4a_beat_block_hammer_frames.png`).

## Phase 4b — closed-loop RoboCasa rollout ✅ (Gate 4b)
- New `simulation/robocasa` base (robosuite master + robocasa 0.2 + MuJoCo 3.2.6, headless EGL);
  kitchen assets fetched at runtime into the mounted volume (rule 8). Base validated on gfx1151
  (`test.py` import + `demo_sim_sanity` render smoke).
- X-WAM adapter `experiments/robocasa_xwam/xwam_policy/deploy_policy.py` implements the shared
  `sim_robocasa.Policy` seam over the extracted in-process `DirectXWAM` (`experiments/xwam_core.py`).
  RoboCasa is single-arm: the denormalized `(Ta, 7)` delta-EE chunk is fed straight into
  robosuite's OSC_POSE composite controller (no IK / motion-planning seam).
- Two ROCm/packaging fixes baked in (reproducible): the xwam layer's `ENV PYTHONPATH=/repos/xwam`
  clobbered the base's `/opt/sim` (re-added in the demos); and the adapter dir was renamed
  `robocasa` → `robocasa_xwam` because `experiments/` on `sys.path` made a dir literally named
  `robocasa` shadow the real pip `robocasa` (namespace package → zero kitchen envs registered).
- Result: `TurnOnSinkFaucet`, full horizon, **10 randomized episodes (seeds 0–9) → 9/10 = 90%**
  success (successful runs early-stop in 5–9 model calls). Per-episode videos + `_result.json`
  → laptop `artifacts/xwam/robocasa_cl10/`.

## Phase 4c — interactive demos (both sims) ✅
- Model-agnostic HTTP/MJPEG interactive servers (sync chunk-replay + real-time async-planner with
  a HOLD-while-thinking view) shipped in each sim base, driven by X-WAM through the `Policy` seam.
- RoboCasa: adapter is a thin single-arm wrapper (same 7-D delta path as closed-loop). Both
  `demo_interactive_robocasa[_rt].sh` SMOKE PASS (load policy, HTTP-triggered rollout, stream +
  save video).
- RoboTwin: added `experiments/robotwin_xwam/xwam_policy` (interactive sibling of the closed-loop
  seam) emitting absolute per-arm EE poses `[T,16]`, plus an additive `action_type` plumb in the
  shared `sim_robotwin` harness (`Policy.action_type`, `Scene.take_action(action_type=...)`,
  `Scene.ee_state_vector()` so the RT server HOLDs the current EE pose, not the qpos vector) —
  default stays `"qpos"` so FastWAM/random are unaffected. The mplib/IK shim is applied lazily on
  first predict (the harness only chdirs into `ROBOTWIN_ROOT` once a scene is built). Both
  `demo_interactive_robotwin[_rt].sh` SMOKE PASS; RT run shows `hold%≈88` (planner delivers EE
  chunks that execute, majority HOLD visualizes per-forward planning latency).

## Next
- Dev full manual repro test, then push `wam-XWAM` → `benchmark` (rule 0.3 approval).
- Post-milestone: ANS efficiency sweep (`action_denoise_steps` vs `sample_steps`) toward
  real-time closed-loop; the ~10s action path is the current control-latency baseline.
