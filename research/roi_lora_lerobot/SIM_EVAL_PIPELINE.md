# Closed-loop LIBERO sim-eval pipeline (MolmoAct2 ROI-pruned, AMD clusters)

The trainable ROI pruning pipeline (see `ROI_TRAINING_PIPELINE.md`) produces
single-pass prunable checkpoints. This doc is the **closed-loop simulation eval**
harness that measures their real task success — and the record of how headless
robot rendering was made to work on AMD datacenter GPUs.

Status: **validated end-to-end on the MI300X cluster.** Full converged sweep = 3 keep
fractions × 5 LIBERO suites = **1500 rollouts** completed; results in
`artifacts/roi_eval/` and report §9.

## 1. What runs
`lerobot-eval` on the fine-tuned checkpoint (`--policy.path`), which restores the
trained `roi_gate` + `roi_mask_token` + ROI config, so the **student gate drives
single-pass token pruning in closed loop** (no teacher, no dual pass). The LIBERO
MuJoCo env steps the robot; the policy observes two cameras (`agentview` +
`robot0_eye_in_hand`), predicts an action chunk, repeat until success/timeout.

Harness scripts (in `scratch/`, mirrored; run inside the eval container):
- `roi_eval_sweep.sh` — dispatches units across GPUs; **resumable at task level**
  (a unit with `eval_info.json` is skipped); optional per-run mask dump.
- `run_roi_eval_detached.sh` — docker launcher (device/user/bind pattern, env fwd).
- `chain_roi_eval.sh` — login-node driver that chains ≤2h SLURM segments until the
  `_ROI_EVAL_DONE` marker (mi300x QOS caps a job at 2h).
- `egl_probe.sh` / `render_bench.sh` — the rendering diagnostics below.

Granularity = (keep_frac × suite × TASK); each unit runs `N_EP` episodes of one
task, so every unit finishes well under the 2h cap and the whole sweep is
resumable + chainable.

## 2. Rendering: the core cluster problem, and its resolution
**GPU (EGL) rendering is impossible on MI300X — proven, not a config bug.**
Inside the eval container on an MI300X node:
```
MUJOCO_GL=egl  →  radeonsi: error: can't create a graphics context on a compute chip
                  EGLError: EGL_BAD_ALLOC at eglCreateContext
```
gfx942 (MI300X) is a **compute-only** die; the Mesa `radeonsi` driver refuses to
create a GL/graphics context on it. This is true for *all* MI300X (compute-class) Instinct nodes
(MI300X/MI325X are compute parts). No driver/env tweak changes it — the standard
practice for headless robot eval on datacenter compute GPUs is **CPU rendering**.

**Resolution — OSMesa (CPU/llvmpipe) software rendering:**
```bash
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
```
Validated: renders correctly, and a fresh end-to-end smoke on the converged
checkpoint returns `pc_success=100`.

**Measured cost (real LIBERO scene, 2 cameras, `libero_object` t0, 1 llvmpipe thread):**
| resolution | sim+render sec/step | steps/s |
|---|---|---|
| 128×128 | 0.135 | 7.4 |
| 256×256 | 0.123 | 8.1 |

Rendering is CPU-bound at ~1 thread/worker, so on the **96-core** EPYC node a
single worker leaves the CPU almost idle. That is the key to throughput ↓.

## 3. Throughput: pack workers per GPU (validated)
Because OSMesa render is CPU (~1 core/worker) and the 6B policy uses only ~12 GB
of the **192 GB** MI300X, multiple rollouts co-reside per GPU. `roi_eval_sweep.sh`
exposes `WPG` (workers-per-GPU); workers round-robin onto `NGPU` GPUs.

Validated `WPG=2` on one MI300X: 2 tasks ran **concurrently** and both returned
100%; 2 units finished in ~206 s vs ~208 s for 1 unit serially → **~2× throughput**
with huge memory headroom. Recommended full-sweep config:
```bash
KEEPS="025 050 075" N_EP=10 NGPU=8 WPG=2 \
  MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa bash roi_eval_sweep.sh
```
`WPG=3` is likely safe too (96 cores / 24 workers = 4 cores each); scale until CPU
render becomes the limit.

## 4. Alternative — Alola graphics GPUs (analyzed, NOT adopted)
Per `CLUSTER_ACCESS_GUIDE.md`, Alola has **graphics-capable** GPUs where EGL *does*
work: Radeon PRO **W7900** (gfx1100, 48 GB, `radeonsi` present, docker + 98T
`/projects` + 171T `/scratch`), Strix-Halo (gfx1151), RDNA4 (gfx1200). Probed
`mlse-alola-b38-ws2` (W7900) — graphics driver present, so GPU EGL rendering is
viable there, unlike MI300X.

**Why we stay on the MI300X cluster anyway (for this workload):**
- **Throughput, not render, is the goal.** Render is only *part* of episode time;
  the 6B policy is the rest, and a W7900 (RDNA3 workstation) is materially slower
  at 6B transformer inference than MI300X. GPU-fast rendering is offset by slower
  inference → no net per-rollout win.
- **Parallelism:** the MI300X cluster = 8× MI300X/node (×WPG); Alola graphics nodes are 1–4
  GPUs each → far less aggregate throughput for a 1500-rollout sweep.
- **Migration cost:** Alola is a different **site** (Markham; storage not synced),
  needs a **gfx1100 ROCm rebuild** of the eval image (ours is gfx942) and a ~24 GB
  weight transfer. High effort, worse throughput.

Keep Alola as the fallback if a *rendering-heavy, low-inference* eval ever needs
true GPU rasterization; a `Dockerfile.strix-gfx1151` base already exists in the
repo to seed that path.

## 5. Reproduce
```bash
# on the cluster login node (survives SSH drop): chains 2h segments to completion
KEEPS="025 050 075" SUITES="libero_spatial libero_object libero_goal libero_10 libero_90" \
  N_EP=10 NGPU=8 WPG=2 nohup bash ~/chain_roi_eval.sh >~/eval_full.log 2>&1 &
# progress
bash ~/progress_check.sh
# aggregate + plots + report §9 (laptop)
python scratch/build_report_update.py
```
Mask-dump viz: add `MASK_DUMP=1` (writes per-step keep-mask NPZs via the
`ROI_MASK_DUMP` hook in `encode_image`), then `make_gate_mask_videos.py`.

## 6. Environment notes / gotchas
- `MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa` are **mandatory** on the MI300X cluster; the
  container defaults to `egl` which fails on compute GPUs.
- LIBERO first import wants an interactive dataset-path prompt → the sweep
  bootstraps `$HOME/.libero/config.yaml` non-interactively once (HOME=/cache is a
  mounted volume, so it persists).
- `RESDIR` is a **container** path (`/outputs/...`), since the host `outputs/` dir
  is bind-mounted at `/outputs`.
- The eval image lives on the compute nodes, not the login node — probe/run docker
  via `srun` on an MI300X node (some nodes may lack the eval image).
- Each unit is a fresh `lerobot-eval` process (reloads the 6B model) — the price of
  task-level resumability; `WPG` packing amortizes it across the idle CPU.
