# Cosmos3-Nano-Policy-DROID (Ryzer WAM package) — ROCm port

Port + runtime analysis of `nvidia/Cosmos3-Nano-Policy-DROID` on **AMD Strix Halo (gfx1151) /
ROCm 7.2.2**. Isolated remote clone `~/Ryzers-cosmos3` on branch `wam-cosmos3-nano`. No
weights/images live on the laptop (rule 3); the checkpoint + dataset slice are fetched into the
container at run time.

**Status:** Phases 1–3 validated on gfx1151 —
- **Phase 1** build: cosmos-framework installed on the ROCm base with the CUDA/TE/natten/flash-attn
  pins never pulled; attention routed to SDPA/AOTriton (`scripts/cosmos3_rocm_patches.py`).
- **Phase 2** smoke: checkpoint loads + one forward runs on ROCm (`scripts/model_smoke.py`).
- **Phase 3** open-loop DROID eval: predicted vs GT action chunks, **overall RMSE 0.202** over 60
  queries (`scripts/openloop_replay.py`).
- **Runtime analysis**: full per-component latency + architecture diagram — see
  [`docs/RUNTIME_ANALYSIS.md`](docs/RUNTIME_ANALYSIS.md).

**Model:** ~15.2B-parameter **Mixture-of-Transformers** world-action model — a `Qwen3-VL-8B` MoT
backbone (understanding + generation experts) + **Wan2.2 VAE** vision tokenizer + action/proprio
adapters, denoised by a UniPC rectified-flow sampler. Predicts joint-position action chunks (32×8)
from a concat camera view (wrist / left / right) + proprio + language. BF16.

**Upstream:** `github.com/NVIDIA/cosmos-framework` @ `0fa3ba4` (policy server + inference).
Weights: `hf.co/nvidia/Cosmos3-Nano-Policy-DROID` (gated). Dataset: `hf.co/datasets/nvidia/Cosmos3-DROID` (gated).

## Runtime at a glance (gfx1151, bf16, num_steps=4)
| path | latency | notes |
|---|---:|---|
| action inference (deployed) | **26.1 s** | 95% is the 15B MoT backbone across 8 CFG net calls |
| &nbsp;└ per denoise step | 6.4 s | 2 net calls (CFG cond/uncond) |
| Wan2.2 VAE encode (vision in) | 0.28 s | one concat conditioning frame |
| Wan2.2 VAE decode (video out) | 253.7 s* | 3D-conv; off the action path — **bf16 hangs on gfx1151** |
| peak VRAM | 33.3 GB | cold first call 75.2 s (autotune) |

\*Video decode is visualization-only and hits a MIOpen conv3d pathology; the fix is a tiny Conv2D
decoder — see `docs/RUNTIME_ANALYSIS.md`.

## Package contents
- `Dockerfile` — `FROM ${BASE_IMAGE}` (ROCm 7.2.2), clone cosmos-framework @ `0fa3ba4`, install
  core deps under a `PIP_CONSTRAINT` that pins the base ROCm torch/numpy (CUDA/TE/natten/flash-attn
  never seen), `assert torch.version.hip`.
- `scripts/`
  - `cosmos3_rocm_patches.py` — SDPA/AOTriton attention backend for gfx1151 (upstream ships
    CUDA-only cudnn/flash2/flash3/natten backends); disables `torch.compile`.
  - `model_smoke.py` — Phase-2 module smoke (load + forward + optional decode).
  - `openloop_replay.py` — Phase-3 open-loop DROID eval (per-dim RMSE/MAE, horizon growth, plots).
  - `cosmos3_fetch_droid.py` — robust gated Cosmos3-DROID slice fetch.
  - `cosmos3_latency_profile.py` — per-component CUDA-event latency profiler → JSON.
  - `cosmos3_render_arch.py` — renders the architecture+latency diagram from that JSON.
  - `cosmos3_videogen.py` — world-model rollout video (two-column GT|pred); decode precision switch.
  - `download_checkpoints.sh` / `download_datasets.sh` / `_hf_common.sh` / `strip_cuda_torch.py`.
- `demos/` — `demo_smoke.sh`, `demo_openloop.sh`, `demo_latency.sh`, `demo_videogen.sh`.
- `docs/` — `RUNTIME_ANALYSIS.md`, `results/`, `UPSTREAM_PIN.{commit,readme}.txt`.
- `assets/` — architecture diagram + open-loop eval plots.
- `adapters/` — (deferred) LIBERO HTTP / OpenPI-WebSocket `Policy` seam for closed-loop.

**Scope:** simulator-free (smoke / forward / DROID open-loop / runtime analysis). Closed-loop
(RoboLab = Isaac Sim, proprietary) is deferred; a ROCm-friendly LIBERO HTTP alternative is noted in
`docs/SCOPING.md`.

**ROCm notes:** attention → SDPA/AOTriton (no natten/flash-attn wheels on gfx1151); avoid
TransformerEngine/Megatron/Apex (training-only); Wan2.2 conv3d **decode** needs fp32 or the tiny
Conv2D decoder on ROCm 7.2.2 (bf16/fp16 conv3d hang; fixed upstream in ROCm ≥7.12).
