# Latte on Strix Halo (gfx1151) — analysis & optimizations

All numbers measured on an AMD Ryzen AI Max+ 395 (Strix Halo, `gfx1151`, Radeon 8060S iGPU)
under ROCm 7.2.2 / torch 2.10, class-conditional `Latte-XL/2` (Sky Timelapse checkpoint,
16 frames @ 256px, latent `4x32x32`). Reproduce with `demos/model_analysis.py`,
`demos/knob_sweep.py`, and the quality gate `patches/opt/parity_test.py`.

## System breakdown (one DiT forward, fp32 baseline)

| Component | Params | Latency (fp32) |
|---|---|---|
| MLP (temporal) | — | 1024 ms |
| MLP (spatial) | — | 1018 ms |
| Attention (spatial, seq P=256) | — | 527 ms |
| Attention (temporal, seq F=16) | — | 478 ms |
| Final layer | — | 1.3 ms |
| **DiT total** | **674 M** (mlp 299M · attn 149M · adaLN/other 223M) | **3443 ms** |

- Latte is a **factorized** DiT: `blocks` alternate spatial self-attn (seq = 256 patches/frame)
  and temporal self-attn (seq = 16 frames), 14 pairs. Attention is `O(P²)+O(F²)`, far cheaper
  than full spatiotemporal `O((P·F)²)`. **MLP dominates** (~2.0 s of the 3.4 s fp32 forward).
- 3786 GFLOP / forward; one full sample = `dit_forward × steps` (all frames denoised jointly)
  + VAE encode/decode. See `artifacts/latte/analysis/system_diagram.png`.

## Knob sweep — DiT forward latency (ms), rows = precision, cols = attention backend

`TORCH_BLAS_PREFER_HIPBLASLT=0` (config default):

| | math | sdpa-default | sdpa-flash | sdpa-mem_efficient |
|---|---|---|---|---|
| fp32 | 4789 | 4782 | ERR | 5697 |
| **fp16** | 569 | **311** | **311** | 312 |
| **bf16** | 355 | **312** | **312** | 312 |

`TORCH_BLAS_PREFER_HIPBLASLT=1`: fp16 math drops to 340 (hipBLASLt speeds the manual GEMM) but
SDPA stays ~314, so **SDPA is the robust best regardless of the BLAS path**.

## Applied optimizations (and validation)

1. **Precision fp32 → fp16/bf16 (autocast).** `3443 ms → 315 ms` per DiT forward (**~11×**).
   Upstream sampling already runs the DiT in fp16 (`use_fp16=True`); we confirm bf16 is equally
   fast (316 ms) and numerically safe on gfx1151.

2. **Attention `math` → SDPA (`attention_mode="flash"`) — with an upstream bug fix.**
   The upstream `flash` branch reshaped the SDPA output `(B,heads,N,head_dim)` straight to
   `(B,N,C)` **without the `.transpose(1,2)`** that the `math` branch applies, interleaving heads
   and sequence. On-device check (`patches/opt/attn_parity_probe.py`): as-is `flash` vs `math`
   **cos-sim = 0.38 (WRONG)**; adding the transpose gives **cos-sim = 1.000000, max|Δ| = 1e-7**.
   The fix ships in `patches/rocm_port.py`. With it, SDPA is an **exact drop-in** for `math` and
   **~1.8× faster** at fp16 (569 → 311 ms).

3. **VAE decode precision & batch.** fp32 SD-VAE decode hits a pathologically slow MIOpen conv
   fallback on gfx1151; run the VAE in **fp16** (per-frame encode ≈ 26 ms). The decoder's MIOpen
   algo choice is **batch-dependent** — decode all frames of a sample in one call (large batch)
   to avoid the slow small-batch path. `demos/_gen_frames.py` and the demos already do both.

4. **DDIM step reduction.** The class configs default to 250 DDPM steps; sampling cost is linear
   in steps, so 250 → 50 DDIM is a ~5× wall-clock cut. (Perceptual-quality lever; validate per
   dataset with FVD.)

### Quality gate — SDPA (`flash`) vs reference (`math`)

Same checkpoint, seed, sampler and 50 DDIM steps, 6 Sky videos, fp16
(`patches/opt/parity_test.py`, `artifacts/latte/parity/parity.json`):

| PSNR (mean / min) | SSIM (mean) | LPIPS (mean) | FVD(math, sdpa) | Gate |
|---|---|---|---|---|
| **59.8 / 56.4 dB** | **0.99998** | **9.5e-5** | **0.08** | **PASS** |

The residual is fp16 fused-vs-unfused accumulation only; the optimized path is quality-neutral.

> Note: per-sample PSNR is only meaningful here **because SDPA is numerically equivalent to
> math**. For genuinely different sampler settings (e.g. step count), diffusion trajectories are
> numerically sensitive and diverge into different-but-valid samples, so use the distributional
> metric stack (`tools/metrics`, FVD/FID/IS) rather than per-sample PSNR.

## Combined effect

fp32+math baseline `4789 ms/forward` → fp16+SDPA `311 ms/forward` = **~15× faster DiT**, quality
neutral. A 16-frame Sky sample at 50 DDIM steps then spends ~16 s in the DiT loop plus VAE
decode, versus minutes for the fp32+math baseline.

## Environment knobs (validated, see `config.yaml`)

`HSA_OVERRIDE_GFX_VERSION=11.5.1`, `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`,
`TORCH_BLAS_PREFER_HIPBLASLT=0`, `MIOPEN_FIND_MODE=FAST` (+ persisted MIOpen/triton caches so the
first-run aotriton/MIOpen kernel compilation — the dominant one-time cost — is paid only once).

Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
