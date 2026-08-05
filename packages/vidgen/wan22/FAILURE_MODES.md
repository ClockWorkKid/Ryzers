<!--
Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
-->

# WAN 2.2 TI2V-5B on Strix Halo — Failure Modes & Rejected Levers

Platform: Strix Halo gfx1151 / Radeon 8060S, ROCm 7.14, torch 2.10.0.
Roofline: ~36 TFLOP/s compute, ~230 GB/s bandwidth, ridge ~157 FLOP/byte.

This records optimization levers we implemented, measured, and **rejected as
defaults**. They stay behind OFF-by-default env flags so anyone can re-probe
them, but the production path (see `config.yaml` "Fast route") does not enable
them. The unifying reason they underperform: **the pipeline is
memory-bandwidth-bound**, so levers that only cut FLOPs or attention math buy
little, and levers that fight the ROCm compiler stack break correctness.

---

## 1. `torch.compile` on the DiT — BROKEN (no speedup + corrupt output)
- Env: `WAN22_TORCH_COMPILE=1` (`WAN22_COMPILE_MODE`, `WAN22_COMPILE_TARGETS`).
- Result (1280x704, 33f, 20 steps): **0 speedup** (269.1 s vs 266.4 s DiT) and a
  **fully corrupt 3.4 KB all-black mp4** (7.5 dB PSNR vs reference).
- Root cause (structural, not a config bug): block-level
  `torch.compile(dynamic=True)` cannot fuse usefully around (a) the custom Triton
  flash-attention op, which is opaque to inductor and forces a graph break, and
  (b) the mandatory fp32 modulation / autocast regions inside each DiT block.
  With inductor's `suppress_errors=True` the failure is swallowed silently and
  the graph emits NaNs → black frames.
- Why we are not chasing a fix: even a clean FFN-only compile is bounded by #4 —
  the DiT is bandwidth-bound, so the achievable ceiling is tiny.
- Status: **disabled by default. Do not ship.**

## 2. DiT token sparsity — training-free K/V ToMe — small, bounded, lossy
- Env: `WAN22_TOME_R=<0..1>` (bipartite soft-match merge of key/value tokens;
  queries kept full-res to preserve 3D RoPE and output length).
- Result: r=0.25 → **−3.0% DiT**, output coherent; r=0.5 → −7.6% DiT but
  **visibly blurry**.
- Why the win is small: merging only K/V cuts attention-score work O(Lq·Lk) but
  leaves the FFN and the activation traffic (the bandwidth-bound majority)
  untouched. Attention is only part of the DiT.
- Status: **off by default.** `WAN22_TOME_R=0.25` is available as an optional
  ~3% knob for the impatient; r>=0.5 is not recommended (detail loss).

## 3. Temporal-delta / keyframe VAE decode — low ROI + architectural blocker
- Idea: decode keyframes fully and reconstruct in-between frames from latent
  deltas to skip decoder work.
- Audit: reference consecutive-frame PSNR = **15.7 dB mean / 12.9 dB min** on a
  high-motion scene → adjacent frames differ a lot, so there is little temporal
  redundancy to exploit.
- Blocker: the VAE decoder uses **causal 3D convs with a `feat_cache` chain** —
  skipping a latent frame corrupts the cache for every subsequent frame, so this
  needs decoder surgery (and likely retraining) rather than a runtime patch.
- Status: **deferred.** Might help static/slow scenes; not worth it for dynamic
  content and not a drop-in runtime lever.

## 4. Whole-video batching for throughput — no gain (bandwidth saturated)
- Idea: generate N videos together (batched DiT + lockstep UniPC + batched VAE)
  to amortize kernel launches and fill the device.
- Result (17f, 10 steps): throughput improvement **≤3%** while cost scales
  near-linearly — DiT 70→135→270 s and VAE 38→75→151 s for N=1→2→4.
- Conclusion: a **single 720p video already saturates the 230 GB/s bus**, so
  batching just multiplies memory traffic. Batching only helps when per-sample
  work is too small to saturate the device (not the case here).
- Status: **serialize (N=1).** Do not batch videos on this hardware.

## 5. CFG cond/uncond batching (B=2) — neutral here
- Env: `WAN22_CFG_BATCH=1` (packs cond+uncond into one B=2 forward for the t2v
  loop).
- Result: neutral (~69→68 s DiT). At ~4400 video tokens each forward already
  saturates the device, so B=2 does not climb the roofline. It also only patches
  the t2v path (no effect on i2v) and the source-patch is fragile.
- Status: **off by default** (kept as an opt-in env flag). Bit-similar, harmless,
  but buys nothing on this model/hardware.

## 6. Sparse 3D conv (SPCONV) for the VAE — not applicable
- Idea: replace dense conv3d with sparse conv to cut VAE compute/bandwidth.
- Rejected on analysis: video latents/activations are **dense** (no spatial
  sparsity to harvest), and the VAE is bandwidth-bound on dense activation
  traffic, not compute-bound. Sparse kernels would add gather/scatter overhead
  for no benefit.
- Status: **not pursued.**

---

## What actually works (the fast route — see `config.yaml`)
The shipped default stack is **lossless**. A full 5 s clip (121f / 50 steps,
1280x704) generates in ~62 min (DiT ~55 min + feathered bf16 tiled VAE ~6.6 min);
shorter clips (105f) land under an hour.

| Lever | Env default | Effect |
|---|---|---|
| bf16 VAE decode/encode | `WAN22_VAE_BF16=1` | VAE 122→40 s (~3x); biggest single lever |
| 12x12 feathered-overlap tiles | `TILE_H=12 TILE_W=12 STRIDE_H=8 STRIDE_W=8` | 4-cell (64px) overlap, cross-faded so tile seams are invisible (see seam fix below) |
| batched tile decode | `VAE_TILE_BATCH=8` | decode 8 tiles per VAE call |
| text-emb + cross-attn K/V cache | `WAN22_TEXT_KV_CACHE=1` | bit-exact; neutral speed here but harmless |
| i2v VAE encode offload | `WAN22_VAE_OFFLOAD=1` | free/offload VAE after pre-loop encode; fixes 121f i2v OOM (see below) |
| fewer sampling steps | `SAMPLE_STEPS` (opt-in) | linear DiT lever (10→8 steps: 68→54 s) |

### Seam fix: no-overlap tiling regressed quality
An earlier fast-route used 8x12 **no-overlap** tiles (`STRIDE==TILE`,
`CROP_MARGIN=0`) for speed. Because each tile was decoded without neighboring
latent context, the VAE's conv padding produced border artifacts at every tile
edge — a visible rectangular "patch" grid in smooth regions (e.g. lighting
gradients). Verified against a matched frame-0 crop: excess gradient energy at
tile boundaries was 1.10x median (seams) vs 0.85x (no seams) after switching to
12x12 tiles with a feather-blended 4-cell overlap. The feather window sums to ~1
across seams and down-weights artifact-prone borders. Net cost: VAE decode
~219→395 s at 121f — acceptable given VAE is a small fraction of total time.

### i2v 121f OOM was NOT a hardware ceiling
Plain-upstream t2v at 121f/50-step (`offload_model=True`) fits and completes on
this machine. Optimized i2v at 121f OOM'd in DiT `rope_apply` even though the DiT
forward is byte-identical between t2v and i2v. Root cause: i2v runs `vae.encode()`
on the full 704x1280 conditioning image right before the denoise loop; those
encoder activations linger in the HIP caching allocator and fragment VRAM,
starving a 320 MB DiT allocation. `WAN22_VAE_OFFLOAD=1` parks the VAE on CPU and
empties the cache after encode, restoring headroom — 121f i2v then completes.

Combined lossless speedup **~6.05x** (17f/10-step bench: 683→113 s); with fewer
steps up to **~7.0x**. After bf16 VAE, VAE (~37 s) and DiT (~54–70 s) are
comparable — remaining single-video latency is DiT self-attention over the video
tokens, whose only further levers are **fewer steps** and **lower precision /
distillation** (i.e. cut bytes moved, not FLOPs).

Grounding: `artifacts/wan22/opt_sweep_results.json`,
`artifacts/wan22/sq_20260804_143002/speedquality_results.md`,
`artifacts/wan22/capacity_20260804_090905/capacity_analysis.md`.
