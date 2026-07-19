# Micro-World runtime optimizations on Strix Halo (ROCm)

A living overview of the inference-time optimizations for **Micro-World** — AMD-AGI's Wan2.1-based,
action-controlled interactive world model (T2W 1.3B + I2W 14B; DiT + action control, conv3d WAN VAE,
T5 text encoder, CLIP image encoder for the image paths) — on **AMD Strix Halo (Radeon 8060S,
`gfx1151`), ROCm 7.2.2, PyTorch 2.10 + bf16**. Levers are grouped into **portable** ones (from
`docs/RUNTIME_OPTIMIZATION_PLAYBOOK.md`, shared across our WAM/VLA ports) and **Micro-World-specific**
ones, with the measured component split that dictates which levers actually move the needle here.

> Method note: numbers are measured on-device with `scripts/opt_ab_microworld.py`, which loads the
> cheap **t2v (Wan2.1-T2V-1.3B)** pipeline once (fits full in the ~47 GB GPU-addressable pool, peak
> ~18 GB) and sweeps
> the levers in-process so every A/B shares the same resident weights (rule 0.5: bounded GPU use —
> one small model, one process). CUDA-event inclusive timers wrap the transformer / VAE-decode / T5
> forwards. Config: 25 frames, 256×448, seed 43, UniPC, 30-step reference. Raw data:
> `artifacts/micro-world/mw_opt_ab.json`. The 14B I2V/I2W paths share the identical DiT/VAE
> architecture (just larger + CPU-offloaded), so the *relative* lever behaviour transfers; only the
> absolute VAE fraction grows with the larger frame count / resolution.

---

## 1. TL;DR — what shipped, and how much

| Optimization | Class | Metric | Gain | Fidelity vs 30-step ref |
|---|---|---|---|---|
| **conv3d override** (`cudnn.enabled=False`) | kernel | WAN VAE decode | **4.63 s → 3.10 s (1.49×)**; total 1.02× | PSNR 26.6 dB (numeric-equiv, bf16 conv algo differs) |
| **TeaCache** (built-in, `threshold=0.10`) | step-skip | full t2v gen | **56.3 s → 37.3 s (1.51×)** | PSNR **23.0 dB** (best-fidelity accel) |
| **TeaCache** (`threshold=0.20`) | step-skip | full t2v gen | 56.3 s → 26.9 s (2.09×) | PSNR 20.3 dB |
| fewer steps (UniPC 30→20) | NFE | full t2v gen | 56.3 s → 38.8 s (1.45×) | PSNR 18.0 dB (coarser sample) |
| fewer steps (UniPC 30→10) | NFE | full t2v gen | 56.3 s → 21.5 s (2.62×) | PSNR 14.8 dB (soft) |
| cross-attn K/V cache | cache | DiT loop | **1.00× (no gain)** | **bit-exact, max\|Δ\|=0.0** |

**Headline / the reframe that matters:** unlike VERA (whose VGGT-DPT + VAE **conv3d** dominated → the
conv override was an 8.8× win), Micro-World's cost is **dominated by the DiT denoise loop — 52.2 s of
56.3 s ≈ 93 %** (VAE 5.5 %, T5 1.8 %). So the real movers here are the **built-in TeaCache** and the
**step count**, not the conv override. The conv override is kept **default-on** because it is free and
its VAE win *grows* on the image paths (I2W decodes 49 frames at 352×640, where the VAE fraction is
much larger than in this 25-frame proxy), but it is not the headline on this DiT-bound model.

Shipped defaults (baked into `demos/mw_generate.py`):
- **conv override** default-on, kill-switch `MW_DISABLE_CUDNN=0` (`apply_gfx1151_speedups()`).
- **TeaCache** default-on with per-mode thresholds — `t2v=0.10`, `t2w=0.20`, `i2v=0.10`, `i2w=0.20`
  (`TEACACHE=0` to disable, `TEACACHE_THRESHOLD` to override).
- **UniPC 30 steps** default (`NUM_STEPS=20` for ~1.5× at solid quality; 10 gets soft).
- 14B/18B I2V/I2W default `GPU_memory_mode=model_cpu_offload` — full residency exceeds the box's
  **~47 GB GPU-addressable memory** (32 GB VRAM carveout + GTT), so T5/CLIP are encoded then evicted
  and the DiT stays resident for the denoise loop (~105 s/step at 49f/352×640, compute-bound).

---

## 2. Component split (where the 56.3 s goes) — t2v-1.3B, 30 steps, conv override on

| Component | GPU time | % of gen | calls | Notes |
|---|---:|---:|---:|---|
| **WAN DiT denoise loop** | **52.2 s** | **92.7 %** | 30 | 1 forward/step, bf16, flash SDPA — dominant, video **self-attention** bound |
| WAN VAE conv3d decode | 3.10 s | 5.5 % | 1 | conv via ATen im2col+GEMM (override); scales with frames×res |
| T5 text encode | 0.99 s | 1.8 % | 1 (+neg) | short, run once per gen |
| peak VRAM | 18.0 GB | — | — | full-resident 1.3B; ample headroom in the ~47 GB GPU-addressable pool |

Because the DiT is **video-self-attention-bound**, levers that attack redundant *cross-attn* compute
(K/V caching) do almost nothing, while levers that remove *whole DiT forwards* (TeaCache step-skip,
fewer steps) scale ~1:1 with the 93 % that actually costs.

---

## 3. Portable levers mapped to Micro-World (from the playbook)

| Lever (playbook §) | Applies? | Priority | Why |
|---|---|---|---|
| **Built-in TeaCache** (§2.1 residual step-skip) | **YES — measured** | **ship (default-on)** | thr 0.10 → **1.51× @ 23.0 dB** (best fidelity/speed); thr 0.20 → 2.09× @ 20.3 dB. Micro-World-specific lever VERA lacked; directly attacks the 93 % DiT loop. Lossy → per-mode thresholds, visually validated (Gate 3). |
| **Fewer denoise steps / distillation** (§2.6) | **YES** | 1 (highest single-gen, lossy) | UniPC 30→20 = 1.45×, 30→10 = 2.62×. PSNR vs 30-step ref (18.0/14.8 dB) is a *divergence* metric — fewer steps is a legitimately coarser sample, not an error; judge visually. 20 steps solid, 10 soft. |
| **conv override** (`cudnn.enabled=False`) (Latte/VERA) | **YES — measured** | ship (small here, grows on image paths) | VAE decode **1.49×** (4.63→3.10 s), total 1.02× on t2v. VERA's 8.8× came from a conv-dominated stack; here the DiT dominates so the win is VAE-local. Free, default-on, kill-switch `MW_DISABLE_CUDNN=0`. |
| **bf16 autocast on hot paths** (X-WAM precision) | **already present** | — | Upstream pipeline already wraps the DiT/VAE in bf16 autocast and weights load bf16 — captured, nothing to add. |
| **Cache cross-attn K/V across denoise steps** (§2.1) | **YES — measured, ~0 gain** | do-not-bake | **bit-exact (max\|Δ\|=0.0)** but **1.00× DiT** — the DiT is video-self-attn bound and the (single-image + T5) cross-attn context is small vs the video tokens. Prototype kept in `scripts/opt_ab_microworld.py` (`kv=True`); not baked (no measurable win). |
| **cfg_skip** (built-in) | candidate | 2 (lossy) | `enable_cfg_skip(ratio, steps)` skips the uncond DiT branch on later steps → attacks the DiT loop. Wired via `CFG_SKIP_RATIO`; not swept here (compounds with TeaCache — validate jointly). |
| scheduler swap (UniPC vs DPM++) | available | low | `SAMPLER=Flow_Unipc`/`Flow_DPM++`; UniPC default (good low-step fidelity). |
| **Increase batch M** (§2.4) / **skip decoder** (§3.2) | N/A here | — | single sequential gen (no free rows for one clip); the decoded pixels *are* the deliverable (can't skip decode). |

---

## 4. Recommended config (ROI order)

1. **Ship TeaCache default-on** at the per-mode thresholds above — biggest safe mover on the DiT loop
   (1.5–2.1×), already wired and Gate-3 validated.
2. **Keep conv override default-on** — free, and its VAE win grows on the 49-frame image paths.
3. **Expose steps** (`NUM_STEPS`): 30 default for quality, 20 for ~1.5× at solid quality.
4. **cfg_skip** as an opt-in further DiT lever for latency-critical runs (validate jointly with TeaCache).
5. cross-attn K/V cache: **not baked** — bit-exact but zero measured gain on this self-attn-bound DiT.

---

## 5. Reproduce / toggles

- **Baked opts (default on):** `MW_DISABLE_CUDNN=1` (conv override, `apply_gfx1151_speedups()` in
  `demos/mw_generate.py`), `TEACACHE=1` with per-mode `TEACACHE_THRESHOLD`. Set to `0` to A/B.
- **A/B + profile harness:** `scripts/opt_ab_microworld.py` (component profile, conv A/B, steps
  sweep, TeaCache A/B, cross-attn K/V A/B). `RUN_SET=conv,steps,teacache,kv` selects blocks;
  `NUM_FRAMES`/`SAMPLE_SIZE`/`SEED` set the workload. Writes `mw_opt_ab.json`.
- Run inside the container from the repo root, e.g. `cd /repos/micro-world && python
  /ryzers/scripts/opt_ab_microworld.py` (mirror `demos/*.sh`).

---

*Scope note: numbers are for Strix Halo `gfx1151` / ROCm 7.2.2 / bf16, measured on the t2v-1.3B proxy
(25 frames, 256×448, UniPC). The 14B image paths share the DiT/VAE architecture; the VAE fraction
(hence the conv-override win) grows with their larger frame count and resolution.*
