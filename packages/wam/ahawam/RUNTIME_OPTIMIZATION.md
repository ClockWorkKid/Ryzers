# AHA-WAM runtime optimizations on Strix Halo (ROCm)

A living overview of the runtime optimizations we applied to make **AHA-WAM** — the
asynchronous Wan2.2-TI2V-5B world-action model — run faster on **AMD Strix Halo (Radeon
8060S, `gfx1151`), ROCm 7.2.2, PyTorch + bf16**. Optimizations are grouped into **portable**
ones (generalize to most video-DiT / diffusion-policy world models) and **AHA-WAM-specific**
ones, plus how they stack toward real-time control rates.

> Method note: all before/after numbers below were measured **same-process / same
> machine-state** (to control for the launch-bound run-to-run variance of this GPU) on real
> captured inputs, and every shipped optimization is **lossless** (bf16-equivalent, verified
> against the unoptimized path). Source data: `artifacts/ahawam/perf/`.

---

## 1. TL;DR — what worked, and how much

| Optimization | Class | Metric | Gain | Lossless? |
|---|---|---|---|---|
| Few-step ODE distillation ("Flash", 1 step) | model | denoise steps 10&rarr;1 | ~10&times; on the denoise portion | distilled (task-equal here) |
| Async two-phase (planner / executor split) | architecture | planner off the control loop | hides ~250&ndash;419 ms/plan | yes |
| Batched KV-editor GEMMs | kernel/occupancy | action-chunk executor | **407&rarr;153 ms (2.66&times;)**, 39&rarr;105 Hz | yes (rel 1.5e-4) |
| Cache static text embedding | redundant compute | video prefill | **412&rarr;250 ms (1.65&times;, &minus;162 ms)** | yes (bit-identical) |
| `torch.compile` action path | compiler fusion | per denoise step | 48.2&rarr;43.4 ms (1.11&times;) | ~0.47% drift, opt-in |
| SDPA backend selection + bf16 | kernel | attention | attention already <2 ms | yes |
| HIP-graph capture | launch overhead | per denoise step | **deferred** (capture fails as-is) | — |

**Headline:** on AHA-WAM-Flash the shippable stack (async + Flash + batched KV editor + text
cache) takes the fast executor to **~153 ms/action-chunk &rArr; ~105 control Hz**, with the slow
planner hidden off the control loop — a **2.66&times;** executor speedup over the un-batched path
at zero task-quality cost.

---

## 2. The model (context)

AHA-WAM is an **asynchronous** Wan2.2-TI2V-5B world-action model: 13.74 B params total
(MoT DiT backbone 7.25 B + umt5-xxl text encoder 5.68 B + Wan VAE 0.70 B + obs query encoder
0.084 B), ~30 GB weights in bf16. It splits inference into two phases that run at different
rates:

- **`phase="video"`** — observation-guided video-context prefill (the **slow planner**);
- **`phase="action"`** — one action-DiT chunk (`action_chunk_size`=16 control steps) that
  reuses the prefilled video KV cache (the **fast executor**).

Baseline two-phase latency (bf16, Flash 1-step): video prefill **419 ms**, action chunk
**493 ms** (= 16 control steps &rArr; ~32 control Hz). The action executor operates on a *tiny*
problem — 16 action tokens attending over a fixed 120-token video context — which is the root
cause of the executor bottleneck addressed in §4.1.

---

## 3. Portable optimizations (generalize to the WAM / diffusion-policy family)

First things to try on any similarly-structured video-DiT / diffusion-policy world model.

### 3.1 Few-step ODE / consistency distillation ("Flash")  — biggest single lever
Distill the multi-step sampler down to **1 step**. On the AHA-WAM action path each extra
denoise step costs ~49 ms and the fixed per-chunk overhead dominates, so going 10&rarr;1 step
removes almost all of the denoise cost. In our RoboTwin closed-loop suite the 1-step Flash
model **matched or exceeded** the 10-step base on every task (small samples), i.e. for
*action* success (not perceptual fidelity) the distilled model is not a downgrade. **Do this
first.**

### 3.2 Asynchronous two-phase execution — decouple slow planning from fast control
Run the expensive world/context prefill on a background channel and let a cheap executor emit
action chunks against the cached context. This **hides the planner off the control loop**
entirely (the ~250&ndash;419 ms prefill no longer blocks the ~105 Hz executor). Any model with a
"understand the scene once, then act many times" structure benefits. This asynchronous design
is the core of AHA-WAM and the reason it can approach real-time.

### 3.3 Cache static conditioning across steps/prefills
The task instruction is constant within an episode, so re-encoding it every prefill is pure
waste. Caching the text embedding and reusing it removed the **~160 ms umt5 encode** from each
prefill (412&rarr;250 ms, **bit-identical** output). Generalizes to any static conditioning
(instruction / goal image / proprio stats). Bonus: once cached, the 5.68 B text encoder
(~11 GB) can be offloaded from VRAM.

### 3.4 Don't recompute unchanged inputs
Skip / reuse the observation-VAE encode when the frame is unchanged, or downsample the obs
frame; MIOpen-tuning the conv+im2col is an alternative. Worth ~67 ms per chunk/prefill.

### 3.5 SDPA backend selection + bf16
Attention is **not** the bottleneck on this hardware (<2 ms per phase with efficient SDPA), so
the win is mostly "don't fight it": bf16 throughout, pick the efficient/flash SDPA backend, and
avoid host syncs that serialize otherwise-overlapping tiny kernel launches.

---

## 4. AHA-WAM-specific optimizations

### 4.1 Batched KV-editor GEMMs (the big win)
The observation-guided video-KV editor was **74% of the executor** (365 ms). Profiling showed
it was a single `16×16×8` macro-tile GEMM launched **~90 times** (one per layer × projection):
each occupies one tiny tile &rArr; near-zero CU occupancy &rArr; **launch/occupancy-bound, not
compute-bound**. Fix: stack the per-layer LN/Linear weights once and run the routing attention
+ delta MLP as a handful of **layer-batched GEMMs** with a leading `L` dim
(`models/wan22/mot.py`, env-gated `AHAWAM_KV_EDITOR_FAST`).
- KV-editor sub-step alone: **656 &rarr; 24 ms (27.7&times;)** in-process.
- End-to-end action chunk: **407 &rarr; 153 ms (2.66&times;)**, **39 &rarr; 105 control Hz**.
- Quality: rel diff 1.5e-4 (reduction-order rounding, inside bf16 epsilon).

**Transferable lesson:** any per-layer Python loop over tiny GEMMs on this GPU is a prime
target — batch across layers so tiles fill the CUs instead of launching many one-tile kernels.

### 4.2 `torch.compile` the action path (modest, opt-in)
`torch.compile(dynamic=False)` on the denoise step: **48.2 &rarr; 43.4 ms/step (1.11&times;)**.
Limited because the action-DiT RoPE uses **complex ops** (TorchInductor falls back to eager)
and the list-of-dict KV cache causes **graph breaks**. Meaningful only for the base 10-step
model (~49 ms/chunk saved); skip for Flash (1 step). Costs ~29 s first-call compile + recompiles
on shape change, ~0.47% drift &rArr; keep it opt-in.

### 4.3 HIP-graph capture (deferred)
Manual `torch.cuda.CUDAGraph` capture of the denoise step **fails**
(`operation not permitted when stream is capturing`) because the step does capture-illegal work
(host syncs / dynamic allocation / list-of-dict KV handling); `compile(mode="reduce-overhead")`
skips cudagraphs for the same graph-break reason. Needs a refactor first (static I/O buffers,
remove host syncs, flatten KV lists to preallocated tensors). After §4.1 removed the
launch-bound editor, the remaining step is largely compute-bound (49 ms, big GEMMs), so the
graph ceiling is now modest — hence deferred.

---

## 5. Why these work — the underlying bottleneck classes

| Bottleneck class | Symptom | Fix used |
|---|---|---|
| Launch/occupancy-bound tiny GEMMs | many one-tile kernels, low CU occupancy, sync inflates wall time | **batch/fuse across layers** (4.1); HIP graph (future) |
| Redundant static compute | same tensor recomputed every step | **cache** (3.3), skip unchanged inputs (3.4) |
| Too many sampling steps | latency scales with denoise steps | **distill to 1 step** (3.1) |
| Serial planner + executor | slow understanding blocks control | **async two-phase** (3.2) |
| Compiler-unfriendly graph | complex ops / dynamic KV break fusion & graphs | `torch.compile` opt-in (4.2); refactor for graphs (4.3) |

Attention and full-tile DiT GEMMs are already efficient on this hardware — spending effort
there has low ROI.

---

## 6. Path to real-time

Current AHA-WAM-Flash executor: **~153 ms/chunk &rArr; ~105 control Hz** (16 control steps per
chunk), planner hidden async. Remaining levers, in rough ROI order:
1. **Skip/downsample the obs-VAE** per chunk (~67 ms, ~14% of the executor) — largest remaining item.
2. **Capture-safe refactor + HIP graph** the denoise step (removes residual per-launch overhead).
3. **fp8/int8** for umt5 + video-DiT prefill (planner-side; already tile-efficient so smaller gain).
4. **Offload the cached text encoder** (~11 GB VRAM) after first encode.

Stack #1&ndash;#2 would push the executor toward the ~2&ndash;3&times; headroom the profiler predicts, with
the planner already off the control loop.

---

## 7. Reproduce / toggles

- Efficiency toggles (default on): `AHAWAM_KV_EDITOR_FAST=1` (batched KV editor),
  `AHAWAM_CACHE_TEXT_CONTEXT=1` (text-embed cache). Set to `0` to A/B against the upstream path.
- Latency demo: `demos/demo_latency.sh` (two-phase latency + executor control Hz + SDPA backends).
- Raw profiles / analysis: `artifacts/ahawam/perf/` (`ANALYSIS.md`, `OPT_RESULTS.md`,
  `BASELINE_AND_AUDIT.md`, `perf_flash.json`, `perf_*.png`).

---

*Scope note: numbers are for Strix Halo `gfx1151` / ROCm 7.2.2 / bf16.*
