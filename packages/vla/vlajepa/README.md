### VLA-JEPA

This package runs [VLA-JEPA](https://github.com/ginwind/VLA-JEPA) (ECCV 2026) on AMD
Ryzen AI Max+ 395 (Strix Halo, gfx1151). VLA-JEPA is a Vision-Language-Action model
(Qwen3-VL VLM + a GR00T flow-matching action head) regularized at training time by a
[V-JEPA2](https://github.com/facebookresearch/vjepa2) latent world model. It is built on
the [starVLA](https://github.com/starVLA/starVLA) codebase.

The image layers the VLA-JEPA inference stack on the default Ryzer ROCm base
(`rocm/pytorch`, torch 2.10.0+rocm7.2.2). Upstream's CUDA-only pins (torch, flash-attn,
deepspeed) are not installed; a build-time guard fails if torch is swapped off ROCm. The
one source change is a ROCm bug fix: Qwen3-VL is hardcoded to `flash_attention_2`
(a CUDA-only wheel) upstream, so we patch it to SDPA (`scripts/apply_rocm_patches.py`).

### Build

```sh
ryzers build vlajepa
ryzers run
```

`ryzers run` with no command is a light ROCm environment check (no weights).
Artifacts are written to `workspace/vlajepa/outputs`.

For faster HuggingFace downloads, set your access token:
`export HF_TOKEN=XXXX....XXXX`

### Full-Model Smoke (capability 1)

Load the real VLA-JEPA checkpoint and run one action prediction on ROCm (first run
downloads the checkpoint + Qwen3-VL-2B + V-JEPA2 encoder into the mounted HF cache):

```sh
ryzers run /ryzers/demo_smoke.sh
```

Pre-fetch just the smoke assets without running:

```sh
ryzers run /ryzers/download_smoke.sh
```

### Closed-loop LIBERO (capability 3)

Closed-loop rollouts step the real MuJoCo LIBERO simulator (headless EGL). This package
layers **on top of** the model-agnostic [`simulation/libero`](../../simulation/libero) base
image — the same seam FastWAM uses. VLA-JEPA plugs into the sim harness through a minimal
`sim_libero.Policy` adapter (`adapters/vlajepa_libero_policy.py`, selected via
`POLICY_FACTORY`); the harness owns the env, rendering and episode loop.

Build the chain (builds the sim base, then this layer FROM it) and run:

```sh
ryzers build libero vlajepa                                  # chain: simulation/libero -> vlajepa
ryzers run /ryzers/demos/demo_closedloop_libero.sh           # batch rollouts -> success_summary.json
SUITE=libero_goal NUM_TASKS=2 NUM_TRIALS=10 \
  ryzers run /ryzers/demos/demo_closedloop_libero.sh
ryzers run /ryzers/demos/demo_interactive_libero.sh          # live browser demo (http://localhost:8080)
ryzers run /ryzers/demos/demo_interactive_libero_rt.sh       # real-time demo, robot HOLDs while planning (:8081)
```

Two live demos (both serve MJPEG + a control page over HTTP; use `ssh -L PORT:localhost:PORT`):
`demo_interactive_libero.sh` runs the sim as fast as inference allows; `demo_interactive_libero_rt.sh`
steps the sim at wall-clock `RT_HZ` (default 20) so the robot visibly pauses ("THINKING") while VLA-JEPA
plans the next action chunk, exposing planner latency. Type an instruction (or use the scene's default) to
run that task; each run also saves a rollout MP4 under `workspace/vlajepa/outputs/interactive[_rt]/`.

Per-task result JSON + rollout MP4s and an aggregate `success_summary.json` land under
`workspace/vlajepa/outputs/closedloop/<TAG>/<SUITE>/`. Preprocessing (180-deg image
rotation, 8-d `[eef_pos, quat2axisangle, gripper_qpos]` state, gripper `1-2*g` remap to
LIBERO `{-1 open,+1 close}`) matches upstream `examples/LIBERO/eval_libero.py`, so success
rates are comparable to the reference deployment.

### Useful Knobs

- `MODEL_REPO` / `CKPT_REL` — VLA-JEPA HF repo + which sub-checkpoint to load
  (default `ginwind/VLA-JEPA` / `LIBERO/checkpoints/VLA-JEPA-LIBERO.pt`).
- `BASE_VLM` / `BASE_ENCODER` — base models the checkpoint config is repointed to
  (`Qwen/Qwen3-VL-2B-Instruct`, `facebook/vjepa2-vitl-fpc64-256`).
- `DTYPE` — `bfloat16` (default), `float16`, or `float32`.
- `HF_TOKEN` — HuggingFace token for gated/faster downloads.
- `ryzers run /ryzers/download.sh` — preload all base assets into the mounted cache.
- Closed-loop: `SUITE`, `NUM_TASKS`, `TASK_ID`, `NUM_TRIALS`, `SEED`, `MAX_STEPS`,
  `REPLAN_STEPS` (default = action chunk), `NUM_STEPS_WAIT` (default 10), `IMAGE_SIZE`
  (0 = native render res), `SAVE_VIDEO` / `NUM_VIDEOS`, `TAG`.

### Roadmap (this package)

- [x] ROCm environment + full-model smoke (capability 1).
- [x] Open-loop replay on LeRobot/LIBERO episodes with GT-vs-pred plots (capability 2).
- [x] Closed-loop LIBERO evaluation in MuJoCo via the `simulation/libero` base (capability 3).
- [x] Live interactive demos (standard + real-time), served over HTTP/MJPEG.
- [ ] LIBERO-Plus perturbation-dimension closed-loop eval.
- [ ] SimplerEnv closed-loop (SAPIEN/Vulkan) — stretch.

Closed-loop LIBERO success (10 trials/task, 10 tasks/suite, seed 1000, gfx1151):
`object` 100/100, `spatial` 100/100, `goal` 97/100, `10 (long)` 100/100 — **397/400 (99.25%)**.

### References

- VLA-JEPA: https://github.com/ginwind/VLA-JEPA (arXiv:2602.10098)
- starVLA: https://github.com/starVLA/starVLA
- V-JEPA2: https://github.com/facebookresearch/vjepa2

Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
