# FlowWAM Ryzers Port — Plan & Status

Target: AMD Ryzen AI Max+ 395 "Strix Halo" (gfx1151), ROCm 7.2.2, `rocm/pytorch` torch base.
Base branch: `origin/benchmark`. Work branch: `wam-flowwam`. Authoritative code + weights live on
the remote Strix Halo machine; the laptop keeps code/doc/artifact mirrors only (rule 3). No
AMD-internal access details are committed (rule 9).

## Milestones (each gated on the previous)
- [x] **P0** Scope upstream, pin commit, set up laptop + remote workspaces.
- [x] **P1** Scaffolded `packages/wam/flowwam/` (Dockerfile, config.yaml, test.py, scripts, demos,
  docs, adapters) reusing fastwam patterns; extended `strip_cuda_torch.py` (torch/numpy/**sapien**
  + defensive flash-attn/apex/cublas) + PIP_CONSTRAINT pin to the robotwin base.
- [x] **P2** Built `flowwam-robotwin` (ryzer_env → robotwin → flowwam) on ROCm 7.2.2; **import + GPU
  env smoke PASSED** on gfx1151: torch 2.10.0+rocm7.2.2 (hip 7.2.53211), Radeon 8060S, GPU matmul,
  diffsynth Wan dual-stream + sapien 3.0.0b1 + cv2/h5py/imageio/accelerate import, and **DiffSynth
  attention resolves to the torch-SDPA fallback** (flash3=flash2=sage=False) with a bf16 Wan-attention
  op verified on-device. No ROCm source patches needed.
- [x] **P3** Weight-download + weight-loaded smoke DONE: Wan2.2-TI2V-5B DiT+VAE + UMT5-XXL text enc
  (under Wan2.1-T2V-1.3B per DiffSynth `redirect_common_files`) + FlowWAM world-model ckpt +
  RoboTwin embodiments. `build_pipeline` loads text 5.68B + dit 5.00B + vae 0.705B + flow_stream
  1.19M = **11.39B params**, all finite, ~13 s. DiT: dim=3072 ffn=14336 layers=30 heads=24
  in/out=48 patch[1,2,2]. SeedVR2 refiner deferred (apex is CUDA-only).
- [x] **P4** Per-module validation (rule 2) DONE (all 4 PASS): UMT5-XXL enc → `(1,512,4096)` bf16;
  Wan2.2 VAE enc+dec **MSE 0.006**; dual-stream DiT forward → rgb+flow finite; RAFT + reversible
  FlowCodec **MAE 0.067 px**. Gotcha: `FlowCodec.encode` returns `(rgb, max_magnitude)`.
- [x] **P5** Open-loop world-model eval on WorldArena `test_dataset` DONE. SAPIEN RobotOnlyRenderer
  validated on ROCm/Vulkan (460 ms/frame). `open_loop_eval.py` (reuses upstream
  `RoboTwinRolloutInferenceDataset` + `rollout_generate`): single-chunk anchored to the real first
  frame reproduces GT scene layout; two-column GT|dream (rule 2.b) + flow strip + 10-episode 3-panel
  `[REAL|FLOW|DREAM]`. Timing: VAE warmup ~525 s on ep1, then ~43–68 s/episode; **decode-dominant**.
  Constraint: input W,H must be divisible by 32 (dual-stream latent parity). Long-horizon
  autoregressive rollout (`wm_autoregressive_eval.py`, **world-model stress test, NOT closed-loop**):
  first chunk faithful (20–48 dB), stepwise drift at each self-anchored hand-off (see PORT_SUMMARY §5).
- [~] **P6** **Closed-loop action policy — STAGED (authored; pending GPU validation).** Correction:
  FlowWAM's genuine closed loop is the *flow-ACTION* pipeline in the full `FlowWAM` repo (IDM action
  expert), **not** an autoregressive world-model rollout. Implemented as the upstream flow-action
  server + `robotwin_policy` on our robotwin base via RoboTwin's stock `eval_policy.py` (no sim
  edits): `demos/demo_closedloop_robotwin.sh`, `download_checkpoints.sh robotwin`. Remaining: rebuild
  image w/ closed-loop layer, validate server on gfx1151, run RoboTwin task(s), record success.
- [ ] **P7** Interactive RoboTwin server (HTTP/MJPEG), mirroring fastwam/xwam interactive demos.
- [ ] **P8** Latency/compute analysis per part + FlowWAM-unique optimizations (bf16 baseline; VAE
  decode is the prime target). Candidate knobs in PORT_SUMMARY §7.
- [ ] **P9** Reproducibility docs, cleanup, laptop mirror, PR/merge into benchmark after manual dev
  test + approval.

## Key decisions
1. **Reuse the FastWAM ROCm playbook wholesale** — shared Wan2.2-TI2V-5B / DiffSynth backbone
   (rule 2.1: patch only for ROCm).
2. **Base on `simulation/robotwin`** — FlowWAM requires SAPIEN robot rendering + RoboTwin
   embodiments; the base already has SAPIEN/Vulkan headless on gfx1151 (rule 0.4: layer, don't
   rebuild a base).
3. **Two upstream repos, one package** — `FlowWAM_WorldArena` (open-loop world model) +
   `FlowWAM` (closed-loop action policy). The closed-loop server uses its repo's `diffsynth` via
   runtime `PYTHONPATH` to avoid clobbering the validated open-loop install.
4. **Defer the SeedVR2 refiner** — apex is CUDA-only; stage-1/world-model gen is the deliverable.
5. **bf16 as the default baseline knob**; diffusion-step count out of scope.

## Corrected open question (was Q1: flow → action)
Resolved: the world-model repo emits video+flow only, but the full `FlowWAM` repo ships the IDM
**action expert** + a RoboTwin policy server/client — that is the genuine closed loop, now staged.
