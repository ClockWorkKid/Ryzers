# X-WAM Ryzer — Implementation Plan

Target: a stable `packages/wam/xwam/` Ryzer running X-WAM on Strix Halo (`gfx1151`,
ROCm 7.2.2), plus a new `packages/simulation/robocasa/` base, culminating in a
`wam-XWAM` PR to `AMDResearch/Ryzers`. Structural template: `packages/wam/fastwam/`.

## Approach: direct PyTorch port (ROCm, not CUDA)

X-WAM is a PyTorch codebase (Wan2.2-TI2V-5B lineage: UMT5-XXL + Wan2.2 VAE + video/depth/
action DiT). We port it **as-is**, swapping the CUDA torch stack for the base image's ROCm
torch, patching only for hardware/ROCm bugs (rule 2.1). The first smoke test is the **full
model loading + one forward pass end-to-end**, not submodules assembled separately.

> Note: this intentionally relaxes local cursorrule #2 (independent module validation),
> mirroring the FastWAM port decision. Flagged to the dev; proceed on confirmation.

Heavy steps (Docker build, multi-GB weight/dataset downloads, closed-loop runs) start only
after plan sign-off. Machine is shared — check `docker ps` and wait for a free GPU (rule 10).

---

## Phase 0 — Workspace & version control (this session)
- [x] Scope upstream X-WAM + Ryzers benchmark + remote strix-halo (see `docs/xwam/SCOPING.md`).
- [ ] Remote `~/Ryzers-benchmark`: branch **`wam-XWAM`** off `benchmark`. No push until dev approval (rule 0.3).
- [ ] Clone X-WAM upstream (pinned `72cfb86`, `--recurse-submodules`) to remote `~/X-WAM-src` for reference.
- [ ] Scaffold `packages/wam/xwam/` + `packages/simulation/robocasa/` skeletons; laptop mirror of docs/skeleton.

## Phase 1 — Build image + full-model smoke (ROCm 7.2.2)
Goal: image builds; the **full X-WAM model** loads on ROCm torch and runs one forward pass.
- Dockerfile `FROM ${BASE_IMAGE}` (rocm/pytorch, torch 2.10+rocm7.2.2). Keep base ROCm torch;
  strip upstream CUDA torch pins; drop hard `flash-attn` (route attention to SDPA/AOTriton on gfx1151).
- Bake X-WAM source at pinned commit; `pip install -r requirements.txt` under `PIP_CONSTRAINT`
  pinning base torch + numpy (so the layer composes on plain and sim bases).
- Build-time guard `assert torch.version.hip`; `test.py` = ROCm torch + GPU + deps sign-of-life.
- **Gate 1:** image builds, ROCm torch confirmed, GPU visible, X-WAM imports + constructs.

## Phase 2 — Forward pass with released weights
- `scripts/download_checkpoints.sh`: fetch `X-WAM-checkpoints` (`wan22_5b`, `pretrained`,
  `robocasa_sft`, `robotwin_sft`) + UMT5-XXL/VAE base into the mounted HF cache (rule 8).
- `scripts/model_smoke.py`: load `robotwin_sft` (or `robocasa_sft`), run one full forward from a
  real observation → produce an action chunk (and optionally an imagined future clip). Capture
  cold/steady latency + VRAM. Validate T5 encode, VAE encode/decode, and DiT paths as sub-checks
  within the same run for parity confidence.
- **Gate 2:** full forward runs on real weights without error; latency/VRAM captured.

## Phase 3 — Open-loop evaluation (X-WAM datasets)
- `scripts/download_datasets.sh`: a few `X-WAM-RoboCasa` / `X-WAM-RoboTwin` episodes.
- `scripts/openloop_replay.py`: replay GT observations, predict action chunks; overlay GT vs
  predicted actions on one plot (rule 2.a), report per-dim normalized MAE. Small plots →
  `/outputs` → laptop `artifacts/xwam/`. Optional video-imagination (GT left, imagined right, rule 2.b).
- **Gate 3:** predicted actions track GT within reasonable error on a handful of episodes.

## Phase 4 — Closed-loop simulation
### 4a. RoboTwin 2.0 (reuse existing `simulation/robotwin` base)
- Add `adapters/xwam_robotwin_policy.py` (Policy seam / `POLICY_FACTORY`) + an X-WAM
  `deploy_policy` bridging to RoboTwin's `eval_policy.py`, reusing FastWAM/AHA-WAM RoboTwin wiring.
- Map X-WAM's broker→server→client eval onto the shared harness (or run its own client against
  `simulation/robotwin`); single-GPU (`num_gpus=1`); headless SAPIEN Vulkan RT.
- **Gate 4a:** ≥1 RoboTwin task rollout completes with success; video + success rate to `/outputs`.
### 4b. RoboCasa (NEW base `simulation/robocasa`)
- New `packages/simulation/robocasa/`: robosuite + robocasa (MuJoCo, EGL headless), assets fetched
  at build/runtime from upstream (rule 8); model-agnostic `Policy` seam mirroring `simulation/libero`.
- `adapters/xwam_robocasa_policy.py`; run ≥1 of the 24 kitchen tasks.
- **Gate 4b:** ≥1 RoboCasa task rollout completes; base validated with `test.py` render smoke on gfx1151.
### 4c. Interactive demos
- HTTP/MJPEG interactive demos for RoboTwin + RoboCasa (mirror fastwam `demo_interactive_*`),
  real-time variants where latency allows.

## Phase 5 — Stabilize, document, PR
- `docs/`: `UPSTREAM_PIN.commit.txt`, `RYZER_REPRODUCTION`, `PORT_SUMMARY`, `RUNTIME_OPTIMIZATION.md`.
- Cleanup; no AMD-internal details in tracked files (rule 9); assets via runtime download (rule 8).
- Dev full manual repro test → push `wam-XWAM` to fork (rule 0.3 approval) → PR to `AMDResearch/Ryzers`.

## Post-milestone (research — efficiency)
- Characterize runtime: per-part latency (UMT5 / VAE / video DiT / depth branch / action head),
  attention backend cost, denoise-step scaling on gfx1151.
- **Asynchronous Noise Sampling (ANS)** is the headline lever: sweep `action_denoise_steps` vs
  `sample_steps`, depth-branch on/off, decoupled inference, toward real-time closed-loop on Strix Halo.
- Compare against FastWAM/AHA-WAM in the shared benchmark to locate current WAM bottlenecks.

---

## Confirmed decisions
1. Package `packages/wam/xwam/` on branch `wam-XWAM` off `benchmark`; new `simulation/robocasa` base.
2. **Direct PyTorch → ROCm port**; full-model smoke first (relaxes rule 2).
3. Reuse `simulation/robotwin`; port RoboCasa fresh.
4. Pin upstream `72cfb86`; weights/datasets via runtime HF download.
5. Heavy steps (build, downloads, closed-loop) start only after plan sign-off.

## Open questions for the dev
- **RoboTwin-first or RoboCasa-first** for the closed-loop milestone? (RoboTwin reuses an existing
  base → faster to a green rollout; RoboCasa is the new port + X-WAM's headline benchmark.)
- Keep X-WAM's **broker→server→client** eval process model, or fold inference into the shared
  `simulation/*` `Policy` adapter seam (as FastWAM/AHA-WAM do)? Hybrid is possible.
- Branch name **`wam-XWAM`** off `benchmark` OK, or prefer a differently-named spinoff?

## Risks / watch-items
- Strix Halo is shared — check `docker ps` before builds; wait for free GPU (rule 10).
- SSH via xsjtema01 tunnel can drop — if lost, ask dev to re-establish, then resume.
- 5B video DiT + depth branch footprint on unified memory; bf16 inference.
- flash-attn 2.8.3 has no gfx1151 build → must fall back to SDPA/AOTriton; validate at smoke time.
- RoboCasa/robosuite is a fresh port: MuJoCo/EGL headless render on gfx1151, asset download, numpy pinning.
- decord/ffmpeg mp4 (RGB + depth) decode on the base image; verify wheels.
- Upstream `pip install` may pull CUDA torch — must constrain to preserve ROCm torch.
