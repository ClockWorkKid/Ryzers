# VLA-JEPA Runtime Optimizations (gfx1151)

Inference-latency optimizations for the VLA-JEPA policy, validated on a Strix Halo `Radeon 8060S`
iGPU (gfx1151, ROCm). Four algorithmically near-lossless speedups cut `predict_action` latency
**~2.3–3.3× per call with no closed-loop task-success regression**.

They ship in `adapters/vlajepa_optim.py` and are applied in place after model load by both
`build_policy()` seams (`vlajepa_libero_policy`, `vlajepa_simplerenv_policy`). **On by default**;
set `VLAJEPA_NO_OPT=1` to fall back to the stock path. Diffusion steps are unchanged.

## What the forward pass does

`predict_action()` = **Qwen3-VL-2B prefill (bf16)** → **flow-matching DiT head (4 Euler steps)**.
The V-JEPA2 encoder + JEPA world-model predictor are training-only (world-model loss in `forward()`)
and never run at inference. Baseline decomposition (single WidowX frame, gfx1151):

| Region | ms | % of 290.8 ms |
|---|---:|---:|
| Qwen3-VL vision tower (conv3d patch-embed ≈ 34 ms) | 64.4 | 22% |
| Qwen3-VL LLM prefill | 80.2 | 28% |
| Qwen3-VL lm_head (150k-vocab logits, **discarded**) | 8.3 | 3% |
| flow-matching DiT head (4 × 31.7 ms) | 134.2 | 46% |

The 155 M-param DiT head costs almost as much as the 2.1 B-param VLM — it runs a launch-bound loop of
tiny GEMMs and (by default) forces fp32 matmuls that bypass the matrix cores.

## The four optimizations

| # | Optimization | Effect | Parity vs stock | Notes |
|---|---|---|---:|---|
| 1 | **bf16 DiT head** | head 134.2 → 21.2 ms (6.3×) | Δ ≈ 0.005 | autocast(bf16) around the head; output cast back to fp32 for numpy |
| 2 | **skip lm_head** | −8 ms, free | Δ = 0 | only the last hidden state is used; the vocab projection is pure waste |
| 3 | **cross-attn K/V cache** | DiT loop −14.5% | Δ = 0 | K/V come from the constant VLM context — recomputed identically every Euler step; memoized per layer per call |
| 4 | **conv3d → matmul patch-embed** | vision 63.4 → 29.2 ms (−54%) | fp32 exact (Δ≈5.6e-6) | the patch-embed Conv3d has kernel==stride==full patch, i.e. it *is* a linear projection; a GEMM avoids the slow MIOpen conv3d path |

Rejected: `torch.compile` (redundant once the head is bf16), `cudnn.benchmark` conv search (no effect),
and enabling hipBLASLt (`TORCH_BLAS_PREFER_HIPBLASLT=1` is a **regression** on gfx1151). Reducing
flow-matching steps 4→2 is lossy (parity Δ = 0.26) and is *not* applied.

Cumulative single-frame SimplerEnv latency: **286 → 130 ms (2.20×)**.

## Closed-loop validation — 200-episode LIBERO A/B

Shipped baseline (`VLAJEPA_NO_OPT=1`) vs the optimized default, four LIBERO suites, 10 tasks × 5 trials
= 200 episodes/path, same seeds/initial states. `predict_action_chunk` timed with `cuda.synchronize()`
on both sides → **model-inference GPU wall-clock** (not MuJoCo sim time). LIBERO feeds two camera views,
so its per-predict cost is heavier than SimplerEnv's single view.

**Task success — preserved:**

| Suite | baseline | optimized |
|---|---:|---:|
| libero_spatial | 50/50 (100%) | 50/50 (100%) |
| libero_object | 50/50 (100%) | 49/50 (98%) |
| libero_goal | 49/50 (98%) | 48/50 (96%) |
| libero_10 (long-horizon) | 47/50 (94%) | 48/50 (96%) |
| **Overall** | **196/200 (98.0%)** | **195/200 (97.5%)** |

**GPU inference wall-clock — 2.3–3.3× per predict:**

| Suite | baseline ms/predict | optimized ms/predict | speedup |
|---|---:|---:|---:|
| libero_spatial | 385 | 165 | 2.33× |
| libero_object | 383 | 167 | 2.30× |
| libero_goal | 390 | 161 | 2.43× |
| libero_10 | 558 | 167 | 3.34× |
| **Total infer wall-clock** | **34.9 min** | **12.5 min** | **2.79×** |

Predict counts match within ~3% per suite, so the wall-clock delta is real compute savings, not shorter
episodes. The optimized path is nearly constant at ~160–167 ms/predict across all suites while the
baseline swings 383→558 ms: the parts the opts remove (fp32 head, lm_head, MIOpen conv3d) are exactly
the ones that scale with prompt/scene complexity, hence the 3.34× on the heaviest suite.

## Toggle

```bash
# default: optimized
ryzers run /ryzers/demos/demo_closedloop_libero.sh
# stock fp32 path for A/B or debugging
VLAJEPA_NO_OPT=1 ryzers run /ryzers/demos/demo_closedloop_libero.sh
```

Each opt is guarded independently in `apply_optimizations()` — a structural mismatch (e.g. a future
Qwen3-VL layout change) skips only that opt and logs it, never breaking inference.
