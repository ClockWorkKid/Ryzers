# VERA runtime optimizations on Strix Halo (ROCm)

A living overview of the runtime optimizations for **VERA** - the two-stage *video-to-action*
policy (WAN-2.1 diffusion video planner "dreams" the future → cotracker extracts motion tracks →
**VGGT-1B Jacobian IDM** inverts them to actions) - on **AMD Strix Halo (Radeon 8060S, `gfx1151`),
ROCm 7.2.2, PyTorch + bf16**. Levers are grouped into **portable** ones (from
`docs/RUNTIME_OPTIMIZATION_PLAYBOOK.md`, shared across our WAM/VLA ports) and **VERA-specific**
ones, with the measured post-optimization bottleneck split that dictates the next move.

> Method note: all before/after numbers are measured on-device (MimicGen `stack_d0`, `gfx1151`,
> bf16, 40 sample steps). The two shipped levers are baked defaults with env kill-switches; both are
> validated closed-loop (10-demo rollout 8/10, consistent with upstream ~94% at n=10). Raw data:
> `artifacts/vera_profile/` (`vera_postopt_profile.json`, `vera_fullstep_profile.json`,
> `vera_forward_profile.json`). Profiler injects the speedups the same way the baked patch does
> (`scripts/patch_gfx1151_speedups.py`).

---

## 1. TL;DR - what shipped, and how much

| Optimization | Class | Metric | Gain | Lossless? |
|---|---|---|---|---|
| **conv override** (`torch.backends.cudnn.enabled=False`) | kernel | closed-loop step | **149.8 s → 17.0 s (8.80×)** | yes (exact - routes conv via ATen im2col+GEMM) |
| **IDM bf16 autocast** (VGGT `compute_jacobian`) | precision | closed-loop step | 17.0 s → **15.5 s (9.66× stacked)** | bf16-equiv (cos 0.99998 vs fp32) |

**Headline:** the shipped stack takes a warm MimicGen closed-loop step from **~150 s → ~15.5 s
(9.66×)** with task success preserved (10-demo `stack_d0`: **8/10**). Both are pure kernel/precision
levers - no change to sample steps, horizon, or any algorithm parameter. Baked into
`start_server_mimicgen.build_policy` via `scripts/patch_gfx1151_speedups.py`; kill-switches
`VERA_DISABLE_CUDNN=0` / `VERA_IDM_BF16=0`.

The conv override is the dominant win because the gfx1151 MIOpen has **no tuned 3D/2D conv solver**
for these shapes → a naive fallback that dominated **both** the WAN VAE conv3d **and** the VGGT DPT
`Conv2d`/`ConvTranspose2d`. Routing every conv through `im2col`/`unfold` + GEMM (rocBLAS/hipBLASLt)
collapses it *and* removes a ~350 s one-time MIOpen autotune.

---

## 2. The model (context) - and two facts that gate the playbook

VERA's deployed MimicGen path per control chunk (10 env steps):
`context frames → WAN-1.3B dream (40-step diffusion, bf16) → Wan2.1 VAE decode → cotracker (2 views)
→ VGGT-1B Jacobian IDM → adaptive Tikhonov control`.

Two structural facts decide which portable levers transfer:

1. **The VAE decode is ON the critical path.** cotracker consumes the *dreamed pixels*, so the
   playbook's single biggest structural win - **"skip the decoder"** (FastWAM 40 % / 6×, X-WAM) - 
   **does not apply to VERA**. VERA must decode every plan.
2. **The stack is planner + tracker + IDM, not a joint action head.** Actions come from a *separate*
   VGGT IDM after the video, so the FastWAM/X-WAM "freeze the video branch, run only action tokens"
   structural knob (§4.1 there) has no direct analog. The transferable structural lever here is
   caching the WAN cross-attn K/V across denoise steps (§3.1 below).

Also confirmed in source (`vera/video_model/algorithms/wan/modules/model.py`, `.../link/wan_pipeline.py`):
WAN runs **bf16 throughout** and the VAE is already bf16 (so X-WAM's `vae_bf16` lever is already
captured); MimicGen uses `skip_text_encoder=True` (no text-encoder to episode-cache); the DiT runs
**one forward per step** (40 calls / 40 steps - guidance is folded, so CFG-branch batching is not an
available lever); attention already dispatches to **flash SDPA**. **No TeaCache** is present (the
loop really does N forwards).

---

## 3. Post-optimization bottleneck split - where the remaining 15.5 s goes

Measured with both baked opts on (`scripts/profile_postopt_mimicgen.py`, CUDA-event inclusive
timers; the torch.profiler self-CUDA table is all-zeros on this ROCm build so events are used):

| Component | GPU time | % of step | calls/step | Notes |
|---|---:|---:|---:|---|
| **WAN DiT dream loop** | **8.71 s** | **56.4 %** | 40 | 1 forward/step, bf16, flash SDPA - **now the dominant cost** |
| WAN VAE (conv3d decode) | 2.19 s | 14.2 % | 16 | bf16, conv via GEMM; on the critical path (tracker needs pixels) |
| **cotracker** (2-view) | 1.93 s | 12.5 % | 2 | VERA-unique; runs on the dreamed video |
| VGGT IDM (Jacobian) | 0.75 s | 4.8 % | 10 | **collapsed** from the pre-opt dominant cost by conv override + bf16 |
| control solve + glue | ~1.87 s | ~12 % | - | Tikhonov `linalg.solve`, context encode, Python/data movement |

**The reframe that matters:** pre-optimization the VGGT IDM (its DPT conv fallback) dominated the
~150 s step; the conv override + bf16 **collapsed the IDM to 0.75 s (5 %)**. So **IDM per-frame
batching is now moot**, and the **WAN DiT 40-step denoise loop is the new dominant cost (56 %)**.
Optimization effort should move to the DiT loop.

---

## 4. Portable levers mapped to VERA (from the playbook)

| Lever (playbook §) | Applies? | Priority | Why |
|---|---|---|---|
| **Cache cross-attn K/V across denoise steps** (§2.1) | **YES - measured** | ship (small) | **A/B: 1.05× on the DiT loop (17.41→16.52 s), bit-exact (max\|Δ\|=0.0)**, 30 blocks patched. Only ~5 % because MimicGen's `skip_text_encoder` context is short (cross-attn K/V is a small GEMM vs the video self-attn that dominates the DiT). Free & lossless → worth baking, but not a big mover. |
| **Fewer denoise steps / distillation** (§2.6; Cosmos3 2×, Nano 3-6×) | **YES** | **1 (highest value, lossy)** | DiT loop scales 1:1 with steps (40 calls/40 steps, no TeaCache). 40→20 ≈ −4.3 s (−28 % of the step) - bigger than the K/V cache. Control tasks are often step-insensitive (Nano/Cosmos precedent) → **validate closed-loop success**. |
| **cotracker bf16 / SDPA** (bf16 precision lever) | **YES** | 3 | 1.93 s (12.5 %), VERA-unique, likely fp32. Quick precision win; validate track fidelity. |
| **Parallel-env batching for eval** (§2.4) | **YES (our iteration)** | 3 | Peak 9.4 GB of 96 GB → batch 4-8 envs; ~Nx faster *suites* (not the robot). |
| **TeaCache-style step-skip** (residual caching) | candidate | 4 (lossy) | None present; adding one skips redundant DiT forwards. Lossy → validate. |
| conv3d→matmul patch-embed (§ VLA-JEPA); MIOpen/AOTriton persisted cache + `MIOPEN_FIND_MODE=FAST` (Latte) | partial | low | conv override already routes convs via GEMM; persisted caches only help cold start. |
| torch.compile / HIP graphs (§4) | marginal | low | playbook: ~1.0-1.1×, not launch-bound here. |
| **Skip decoder** (§3.2) · **VAE bf16** (X-WAM) · **CFG-branch batching** (§2.4) · **text-encode cache** (§2.2) · **IDM per-frame batching** | **N/A / already done / moot** | - | decode required; VAE already bf16; 1 forward/step; text encoder skipped; IDM now 5 %. |

---

## 5. Recommended next steps (ROI order)

1. **Sample-step sensitivity sweep (40 → 20 → 10)** with closed-loop success scoring - the DiT loop
   is now dominant and scales 1:1 with steps, so this is the biggest remaining single-plan lever
   (40→20 ≈ −28 % of the step). Lossy → gate on `stack_d0` task success, not just latency.
2. **WAN cross-attn K/V cache** - **prototyped & measured: 1.05× on the DiT, bit-exact**
   (`scripts/crossattn_kv_ab_mimicgen.py`, `artifacts/vera_profile/vera_crossattn_kv_ab.json`).
   Small but free/lossless - bake it as a default-on micro-win alongside the step reduction.
3. **cotracker bf16 autocast** - quick precision win on the 12.5 % tracker; validate track fidelity.

> Prototyped lever result (2-dream A/B, both baked opts on, `stack_d0`, 40 steps): cross-attn K/V
> cache = **1.05× DiT, max\|Δ\|=0.0 (bit-exact)**. Confirms the lever is safe but that VERA's DiT
> cost is dominated by the video **self-attention**, not the (short, text-encoder-skipped) cross-attn
> - so the step count, not K/V recompute, is where the DiT time actually is.

---

## 6. Reproduce / toggles

- **Baked opts (default on):** `VERA_DISABLE_CUDNN=1` (conv override), `VERA_IDM_BF16=1` (IDM bf16
  autocast), applied by `scripts/patch_gfx1151_speedups.py` at build. Set either to `0` to A/B.
- **A/B harness:** `scripts/speedup_ab_mimicgen.py` (per-lever latency + output fidelity).
- **Post-opt component profile:** `scripts/profile_postopt_mimicgen.py`
  (`agent_scripts/vera_run_postopt_profile.sh`) → `vera_postopt_profile.json`.
- **Closed-loop validation:** `agent_scripts/vera_validate_speedup.sh`
  (`NUM_DEMOS`, `ROLLOUT_HORIZON`, `SAMPLE_STEPS`).

---

*Scope note: numbers are for Strix Halo `gfx1151` / ROCm 7.2.2 / bf16, MimicGen `stack_d0`.*
