# VERA Ryzer - Scoping Notes

Scope snapshot captured during initial project scoping. VERA is added as a slim model package on
the `benchmark` branch (models are consumers of shared bases; weights/images are fetched at
runtime, not tracked).

## 1. Goal

Port **VERA** (Video-to-Embodied Robot Action model) into an AMD **Ryzer** package that runs on the
**Strix Halo** platform (Ryzen AI Max+ 395, `gfx1151`, ROCm 7.2.2) as a **direct PyTorch → ROCm
port** - upstream code on ROCm torch instead of CUDA torch 2.6, patching only for hardware/ROCm/
version bugs (rule 2.1). Milestone chain:

1. Build image + **full package import smoke** on ROCm 7.2.2.
2. **Download weights + load a checkpoint** (released checkpoints + frozen upstream bases).
3. **Video generation** - run the author demos, produce "dreamed" future video.
4. **Closed-loop control** - MimicGen (robosuite/MuJoCo) + PushT (gym-pusht) via the server/
   controller/viewer, then the interactive MJPEG viewer.

Final deliverable: a `wam-vera` branch on the fork `ClockWorkKid/Ryzers` adding
`packages/wam/vera/`, PR upstream to `AMDResearch/Ryzers`.

## 2. Upstream VERA (github.com/sizhe-li/VERA, pinned `fe8561aaa3783230d01d696987992f7eeacb2e12`)

Paper: *Turning Video Models into Generalist Robot Policies* (Li et al., MIT 2026). License
**MIT**. Project page vera.csail.mit.edu. HF org `sizhe-lester-li`.

### What it is
A **two-stage, closed-loop, video-to-action policy**. It leaves a video generative model **as-is**
as an action-free world model that "dreams" the future, and trains an **embodiment-specific
inverse-dynamics model (IDM)** built on the robot **Jacobian** to translate the dream into
low-level actions:

1. **Video planner** (`vera.video_model`) - action-free diffusion video model that generates future
   frames from the current observation (+ optional text). **Embodiment-agnostic.** Two backbones:
   **WAN** (Wan2.1, DiT) for MimicGen/DROID/OMNI, and a tiny **DFoT** U-Net3D flow planner
   (~2.4M params) for PushT.
2. **Jacobian IDM** (`vera.idm` + `vera.policy`) - data-efficient dream→actions translator on the
   **VGGT-1B** visual backbone. **Embodiment-specific**, swappable without retraining the planner.

Closed-loop: context frames → planner rolls out a short visual plan → Jacobian IDM inverts each
step into an action chunk → chunk executed → new observations return to the planner.

### Package layout (`vera/`)
```
vera/main.py                    # Hydra entry point (train + eval)
vera/video_model/               # WAN + DFoT video planners (the "dream")
vera/idm/                       # Jacobian inverse-dynamics model (VGGT backbone) + dfot
vera/policy/                    # closed-loop policy wrappers (motion_policy_*)
vera/server/                    # websocket policy server + live viewer
   start_vera_server.py         #   unified launcher (--embodiment {pusht,mimicgen,allegro,droid})
   vis_server.py, save_vis_video.py
vera/controller/  vera/datasets/  vera/env_runner/  vera/configurations/  vera/utils/
examples/{droid_generation,mimicgen_stack,pusht_dfot_stack}.ipynb   # client notebooks
```

### Dependencies (pyproject.toml)
- **Core**: `torch==2.6.0` (pinned for flash-attn ABI - **must strip for ROCm**), numpy, omegaconf,
  hydra-core, einops, jaxtyping, dacite, huggingface-hub, websockets, msgpack, timm, pillow, roma,
  `torchmetrics==1.4.0.post0`, torch-fidelity, mediapy, `setuptools<81`, plotly, easydict.
- **`video`** (WAN planner): torchvision, `pytorch-lightning>=2.5`, `transformers>=4.51`,
  `diffusers>=0.34`, `accelerate>=1.1`, `deepspeed>=0.15`, opencv-python, decord, imageio(+ffmpeg),
  av, `zarr>=3.0`, pandas, scikit-learn, pyzmq.
- **`idm`** (Jacobian IDM): scipy, transforms3d, h5py, moviepy, pytorchvideo, rerun-sdk, matplotlib,
  openai-clip, rotary-embedding-torch, **`vggt @ git+https://github.com/facebookresearch/vggt.git`**.
- **`eval`** (sims): `gymnasium==0.29.1`, `gym-pusht==0.1.5`, `pymunk<7`, `robomimic==0.5.0`,
  `robosuite==1.4.1`, `mimicgen==1.0.0`, `mujoco==3.5.0`.

### Released assets (HuggingFace `sizhe-lester-li/VERA`)
Only **trained** artifacts are hosted; frozen upstream pieces pulled from their homes.

| Group | dir | what | size |
|---|---|---|---|
| MimicGen | `mimicgen-wan-1.3b/` | WAN planner DiT bf16 + `flow_decoder.ckpt` + `algo_config.yaml` | ~2.8 GB |
| MimicGen | `idm-mimicgen-285ouq1q/` | Jacobian IDM (VGGT-1B) for `stack_d0` | ~11.3 GB |
| PushT | `pusht-dfot/` | DFoT U-Net3D flow planner (~2.4M) + `run_config.yaml` | ~39 MB |
| PushT | `pusht-idm/` | PushT Jacobian IDM (~34.8M, 2-DOF) + `config.yaml` | ~232 MB |
| DROID | `wan-droid-14b/` | DROID WAN planner (14B i2v→v2v, 3-view) DiT bf16 + `algo_config.yaml` | ~31 GB |
| OMNI | `omni-wan/`, `idm-*/` | cross-embodiment WAN planner (14B) + IDMs | ~33 GB |

**Wave-1 download (MimicGen + PushT) ≈ 15 GB**; full repo ≈ 73 GB. Frozen bases (not re-hosted):
`Wan-AI/Wan2.1-T2V-1.3B` (MimicGen), `Wan-AI/Wan2.1-I2V-14B-480P` (DROID/OMNI), `facebook/VGGT-1B`.

### The three author demos
| Task | Runs | Client | needs |
|---|---|---|---|
| **DROID video generation** | direct, no server/sim | `examples/droid_generation.ipynb` | DROID WAN-14B + Wan2.1-I2V-14B; ~60 GB VRAM bf16 |
| **MimicGen** 2-block stack | server + client + sim | `examples/mimicgen_stack.ipynb` / `run_mimicgen_eval` | WAN-1.3B + Wan2.1-T2V-1.3B + VGGT IDM + `eval` (MuJoCo/robosuite) |
| **PushT** planar push | server + client + sim | `examples/pusht_dfot_stack.ipynb` | DFoT planner + PushT IDM + `eval` (gym-pusht) + PushT zarr replay |

Server/client pattern: `python -m vera.server.start_vera_server --embodiment X --port ...
--vis-port ...`; `--vis-port` opens a live viewer dashboard (obs | dream+tracks | dream | Jacobian).

## 3. ROCm porting concerns (touchpoints)
1. **Strip `torch==2.6.0` (+torchvision)** from pyproject; keep base ROCm torch 2.10 via
   `PIP_CONSTRAINT` (FastWAM `strip_cuda_torch.py` pattern).
2. **flash-attn**: optional - WAN path falls back to **SDPA** if absent → do not install; rely on
   ROCm SDPA/AOTriton.
3. **WAN (Wan2.1) VAE conv3d decode** on gfx1151 - watch the decode bottleneck; reuse the tuned env
   (AOTriton on, hipBLASLt off). (In practice Wan2.1 bf16 decode did not hang.)
4. **VGGT** git dep - pip-installable public repo; weights `facebook/VGGT-1B` on first use.
5. **numpy** - VGGT requires `numpy<2`; leave unpinned so the resolver settles at 1.26.4 (ROCm
   torch 2.10 round-trips cleanly).
6. **eval extra**: `mimicgen==1.0.0` on PyPI is an unrelated stub → install NVlabs mimicgen from
   source; `robomimic==0.5.0` is GitHub-only and hard-pins an old diffusion/LM stack (unpin the
   three shared deps, install `--no-deps`); offscreen MuJoCo needs EGL/OSMesa GL libs.
7. **VRAM**: DROID 14B gen ≈ 60 GB bf16 - Strix Halo unified memory must be sized for it; the
   MimicGen WAN-1.3B planner is the far lighter target and leads.

## 4. Remote target state (as scoped)
- Target: a Strix Halo (Ryzen AI Max+ 395, `gfx1151`) machine, ROCm 7.2.2, ample free disk. Shared
  machine - GPU checked free before builds (rule 10). ryzers default init image is
  `rocm/pytorch:rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0`.
- A fork clone tracks branch `benchmark` (origin `ClockWorkKid/Ryzers`, upstream
  `AMDResearch/Ryzers`); VERA gets its own `wam-vera` branch off `benchmark`.
- Prior Wan-on-ROCm porting work (FastWAM / Cosmos3 / X-WAM) is directly reusable - VERA's WAN
  planner shares the Wan2.1/2.2 DiT+VAE lineage.

## 5. Key decisions / assumptions
- Package `packages/wam/vera/` (video planner + action IDM), mirroring the `wam/fastwam/` template.
  Branch `wam-vera` cut from `benchmark`.
- **Direct PyTorch → ROCm port**; full import → load ckpt → run author demo end-to-end (relaxes
  rule 2, as FastWAM/X-WAM did).
- Pin upstream `fe8561a`; weights/bases via runtime HF download (rule 8) into a mounted cache.
- Image installs `.[idm,video,eval]`; assets and the frozen bases fetched at runtime.
