# VERA Ryzer — Port Summary (progress log)

Upstream: github.com/sizhe-li/VERA @ `fe8561a` · bases: Wan2.1-T2V-1.3B / Wan2.1-I2V-14B-480P /
VGGT-1B · target: AMD Strix Halo (Ryzen AI Max+ 395, `gfx1151`), ROCm 7.2.2 · branch `wam-vera`
off `benchmark`.

## Phase 1 — image build + full package import smoke ✅ (Gate 1)
- Direct PyTorch → ROCm port: base image ROCm torch preserved; the upstream `torch==2.6.0` CUDA pin
  (ABI-tied to flash-attn wheels) is stripped via `scripts/strip_cuda_torch.py`, everything else
  installed under `PIP_CONSTRAINT`. flash-attn is not installed → the WAN planner falls back to
  SDPA/AOTriton on gfx1151. `numpy` is intentionally left unpinned (the VGGT IDM backbone requires
  `numpy<2`; ROCm torch 2.10 round-trips cleanly on 1.26.4).
- Result: `vera:latest` builds; `test.py` PASS on `Radeon 8060S` (torch 2.10.0+rocm7.2.2,
  transformers 5.13, diffusers 0.39); full `vera{,.policy,.idm,.server,.video_model,.controller}` +
  vggt import.

## Phase 2 — load a released checkpoint ✅ (Gate 2)
- Weights fetched at runtime (rule 8): `scripts/download_checkpoints.sh wave1` (MimicGen 1.3B +
  PushT) / `droid` (14B). Frozen bases auto-pull on first run.
- DROID WAN-14B planner (`WanPipeline.from_config`, bf16) loads in **54 s**, **46.9 GB VRAM** (of
  96 GB); context 29 frames → 24 generated/call.

## Phase 3 — video generation ✅ (Gate 3)
- `examples/droid_generation.ipynb` ported to headless `scripts/videogen_droid.py` +
  `demos/demo_videogen.sh`; one scene-2 prompt generated a 53-frame clip (29 ctx + 24 dream) that
  follows the prompt. **Wan2.1 VAE conv3d bf16 decode did NOT hang** on gfx1151.
- `scripts/videogen_image2video.py` + `demos/demo_img2vid.sh` (held input left / generated rollout
  right, rule 2.b) and `scripts/videogen_multiprompt.py` (fixed context, N language commands, fixed
  seed) added; artifacts mirrored small to the laptop.
- **Per-component profile** (`scripts/profile_videogen.py`, CUDA-event forward hooks + FlopCounter):
  one 29→24 generation, 40 denoise steps — **WanModel DiT 889 s (67.7% wall, 99.6% FLOPs)**,
  **WanVAE encode+decode ~423 s (32.2% wall, conv3d memory/kernel-bound)**, T5/CLIP trivial. Wall
  ~1313 s/gen, peak 49.2 GB. Levers: fewer denoise steps / shorter horizon, DiT attention kernel,
  faster conv3d decode.

## Phase 4 — closed-loop control + interactive viewer ✅ (see `CLOSEDLOOP.md`)
- Rebuilt to `.[idm,video,eval]` + EGL/OSMesa GL libs for headless MuJoCo. VERA source patches are
  applied as a **cheap late Docker layer** (editable install), so patch iterations never re-trigger
  the dependency reinstall.
- **PushT closed-loop ✅** — `demos/demo_pusht.sh` + headless `scripts/closedloop_pusht.py` (ported
  from `examples/pusht_dfot_stack.ipynb`): server (DFoT planner + Jacobian IDM, no MuJoCo) + client
  + `save_vis_video`. 3 rollouts from replay state 3664 (horizon 200, seed 42): **100% success,
  100% mean max-reward, 82% mean coverage**.
- **MimicGen closed-loop ✅** — `demos/demo_mimicgen.sh` drives robosuite/MuJoCo `stack_d0` via the
  upstream `run_mimicgen_eval` controller + WAN-1.3B planner + MimicGen Jacobian IDM, rendering
  headless under **EGL**. A short `NUM_DEMOS=1 ROLLOUT_HORIZON=50` run completed all 5 chunks (env
  stepping `1→11→21→31→41→50`) and dumped `mimicgen_vis.mp4` + per-view episode videos. Latency:
  cold start ~342–350 s (incl. one-time cotracker `torch.hub` fetch + loads), warm **~142 s/chunk**
  (10 env steps).
- **Port fixes (rule 2.1)**, all reproducible in the image build (`scripts/patch_eval_imports.py`):
  1. **NVlabs mimicgen, not the PyPI stub** — the PyPI project named `mimicgen` is unrelated (no
     `mimicgen.utils`); VERA's runner needs `mimicgen.utils.robomimic_utils.create_env`, so the
     Dockerfile unpins it and installs `NVlabs/mimicgen` from source `--no-deps` (+ `gdown`).
  2. **torchmetrics 1.4.x LPIPS import guard** — `NoTrainLpips` / `_valid_img` moved/renamed;
     guarded so the eval-only metrics never hard-fail the policy import (off the inference path).
  3. **Tracker backend field-name bug** — `tracker_backend_from_cfg` read `cfg.tracker_backend`,
     but the WAN pipeline's `MotionTrackConfig` names the field `backend`, so `backend="cotracker"`
     was silently ignored and the tracker always fell back to the vendored-only **alltracker**
     (which VERA's own `build_policy` warns gives wrong-direction flow on MimicGen). Patch honors
     `.backend`; the demo layers `tracker.backend: cotracker` onto a *copy* of the shipped
     `algo_config.yaml`. cotracker is `torch.hub`-fetched at runtime — no vendored binary.
  4. **cotracker visualization dtype+device** — `draw_pts_gpu` runs on `rgbs.device` and
     boolean-indexes the visibility mask; the cotracker wrapper passed float visibility on CPU while
     the video stayed on GPU (→ `IndexError` then a cuda/cpu mismatch). Patch thresholds visibility
     to a bool mask on the video's device and hands `draw_pts_gpu` the GPU tracks.
- A non-fatal `libGLU.so.0` EGL warning fires at MimicGen env-close (cosmetic; offscreen render + all
  video dumps succeed) — add `libglu1-mesa` at the next rebuild to silence it.

## Next
- Dev full manual repro test → push `wam-vera` → `benchmark` (rule 0.3 approval).
- Post-milestone (efficiency): the per-chunk cost is planner-bound (WAN DiT denoise + Wan2.1 VAE
  conv3d) — sweep `--sample-steps` / horizon, attention kernel, and a bf16/decode pass toward
  faster closed-loop control on Strix Halo.
