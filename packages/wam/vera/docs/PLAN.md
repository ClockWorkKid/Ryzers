# VERA Ryzer - Implementation Plan

Target: a stable `packages/wam/vera/` Ryzer running VERA on Strix Halo (`gfx1151`, ROCm 7.2.2),
able to load the released models and run the author demos - video generation and closed-loop
control for both embodiments - culminating in a `wam-vera` PR to `AMDResearch/Ryzers`. Structural
template: `packages/wam/fastwam/`.

## Approach: direct PyTorch → ROCm port (not CUDA, no re-implementation)

VERA is a clean PyTorch codebase (torch 2.6 / CUDA upstream). We port it **as-is**, swapping the
pinned CUDA torch for the base image's ROCm torch 2.10, patching only for hardware/ROCm/version
bugs (rule 2.1). First smoke is **full package import**, then **load a checkpoint**, then **one
author demo end-to-end** - not submodules assembled separately.

> Note: this relaxes local cursorrule #2 (independent module validation), mirroring the FastWAM /
> X-WAM port decisions. Flagged to the dev; proceeded on confirmation.

Heavy steps (Docker build, multi-GB weight downloads, closed-loop runs) start after plan sign-off.
The machine is shared - check `docker ps` and wait for a free GPU (rule 10).

---

## Phase 0 - Workspace & version control
- [x] Scope upstream VERA + Ryzers conventions + the remote Strix Halo target (see `SCOPING.md`).
- [x] Dedicated fork clone; branch **`wam-vera`** off `benchmark`; `packages/wam/vera/` staged.

## Phase 1 - Build image + full package import smoke (ROCm 7.2.2) ✅
- Dockerfile `FROM ${BASE_IMAGE}` (rocm/pytorch torch 2.10+rocm7.2.2, py3.12). Git clone VERA at
  pinned `fe8561a`; `strip_cuda_torch.py` removes the `torch==2.6.0`/torchvision pins; install
  `.[idm,video]` under `PIP_CONSTRAINT` pinning base torch/torchvision. No flash-attn (SDPA
  fallback); VGGT installs from its git dep. numpy left unpinned (VGGT needs `numpy<2`).
- **Gate 1:** image builds; `test.py` green (ROCm torch, gfx1151 GPU matmul, full VERA import).

## Phase 2 - Download weights + load a checkpoint ✅
- `scripts/download_checkpoints.sh [wave1|droid|all]` into the mounted cache (rule 8). Frozen bases
  (`Wan-AI/Wan2.1-T2V-1.3B`, `Wan-AI/Wan2.1-I2V-14B-480P`, `facebook/VGGT-1B`) pull on first use.
- **Gate 2:** a released checkpoint loads and runs a forward pass on gfx1151 (DROID 14B: 54 s load,
  46.9 GB VRAM).

## Phase 3 - Video generation ✅
- MimicGen WAN-1.3B and DROID 14B language-conditioned generation ported to headless
  `scripts/videogen_*.py` + `demos/demo_videogen.sh` / `demo_img2vid.sh`. Context frames left /
  generated future right (rule 2.b). Per-component profile via `scripts/profile_videogen.py`.
- **Gate 3:** generated future video for at least one embodiment; DROID prompt-conditioning visibly
  changes the generation. (DiT dominates wall+FLOPs; Wan2.1 VAE conv3d dominates inefficiency.)

## Phase 4 - Full demos, closed-loop, stabilize ✅
- Add the `.[eval]` sim extra (+ EGL/OSMesa GL libs) → MimicGen `stack_d0` + PushT closed-loop via
  `start_vera_server` + the upstream controller/notebook clients; live MJPEG viewer dashboard.
- **Gate 4a (PushT):** ✅ closed-loop push, 3 rollouts → 100% success, 82% coverage; viewer +
  episode videos.
- **Gate 4b (MimicGen):** ✅ robosuite/MuJoCo `stack_d0` steps closed-loop offscreen under EGL
  (WAN dream → cotracker tracks → Jacobian IDM → action chunk); short rollout + viewer mp4.
- Runtime characterization (planner DiT denoise / Wan VAE decode / VGGT IDM); cold ~342-350 s, warm
  ~142 s/chunk. Docs: `UPSTREAM_PIN.commit.txt`, `PORT_SUMMARY.md`, `CLOSEDLOOP.md`.
- Cleanup; no AMD-internal details in tracked files (rule 9); assets via runtime download (rule 8).
- Dev full manual repro test → push `wam-vera` to fork (rule 0.3) → PR to `AMDResearch/Ryzers`.

---

## Confirmed decisions
1. New package `packages/wam/vera/`; branch `wam-vera` off `benchmark`.
2. **Direct PyTorch → ROCm port**; full import + load-ckpt + demo smoke (relaxes rule 2).
3. Pin upstream `fe8561a`; weights/bases via runtime HF download (rule 8).
4. Image installs `.[idm,video,eval]`: planner+IDM video path + the MuJoCo/robosuite sim stacks.

## Risks / watch-items
- Strix Halo is shared - check `docker ps` before builds; wait for a free GPU (rule 10).
- WAN (Wan2.1) VAE conv3d decode on gfx1151 - watch the decode bottleneck; tuned env is AOTriton on
  (`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`), hipBLASLt off (`TORCH_BLAS_PREFER_HIPBLASLT=0`).
- `torch==2.6.0` hard pin - must be stripped or the install pulls CUDA wheels over ROCm torch.
- Offscreen MuJoCo GL on gfx1151 - EGL first, OSMesa software fallback wired in.
- The eval extra's `mimicgen==1.0.0` (PyPI) is an unrelated stub - install NVlabs mimicgen from
  source; `robomimic==0.5.0` is GitHub-only and pins an old diffusion/LM stack (unpinned for the
  three shared deps, installed `--no-deps`).
