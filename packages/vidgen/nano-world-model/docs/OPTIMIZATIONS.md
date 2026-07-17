# Nano-World-Model inference optimizations — log & outcomes

Canonical record of the NanoWM inference optimizations for AMD Strix Halo (Ryzen AI Max 395+,
gfx1151, ROCm PyTorch 2.10). Companion to `OPTIMIZATION_SCOPING.md` (the ranked opportunity map).
All optimizations are inference-only and live as an override in
`packages/vidgen/nano-world-model/patches/opt/` (`nanowm.py`, `rollout.py`, + test/compose scripts),
gated so the upstream fp32 full-window path is preserved byte-for-byte.

Target workload: sliding-window diffusion-forcing rollout (one frame generated per window, 50 DDIM
steps, sequential schedule, causal temporal attention). The DiT sampling loop is >98% of wall-clock.

## Optimization catalog

| ID | Name | Kind | Status |
|----|------|------|--------|
| A1 | Causal window trim (generate `history+1` frames, not `model.num_frames`) | exact | ✅ round 1 |
| A3 | Precompute RoPE cos/sin once (was recomputed every attention call) | exact | ✅ round 1 |
| A4 | Cache action embedding once per window (was re-embedded every step) | exact (bit) | ✅ round 1 |
| B1 | bf16 autocast on the DiT sampling loop (VAE stays fp32) | numerical | ✅ round 1 |
| A2 | Temporal adaLN de-duplication (drop 255/256 redundant `Linear(D→6D)` rows) | exact | ✅ round 2 |
| C1 | Fewer sampling steps / faster sampler (DPM-Solver++) | numerical | ✅ round 3 |
| A5 | Context-frame KV cache across sampling steps | exact | ✅ round 4 |
| D1 | `torch.compile` the DiT forward (kernel fusion) | fusion | ✅ round 5 |

---

## Round 1 — A1 + A3 + A4 + B1 (implemented & validated)

### Correctness / exactness (modified vs UNMODIFIED upstream, identical weights, same input)

| Check | pusht B/2 (max abs) | csgo L/2 (max abs) |
|---|---|---|
| A3 full-window fp32 vs original | 3.0e-6 | 5.9e-6 |
| A1 trimmed kept-frame vs original[:,h] | 2.1e-6 | 3.7e-6 |
| A4 precomputed action_emb vs in-forward | **0.0 (bit-exact)** | **0.0 (bit-exact)** |
| B1 bf16 vs fp32 (per-forward rel err) | 2.3e-3 | 1.1e-2 |

A1/A3/A4 differ only by floating-point reduction-order noise (~1e-6) → treated as exact.

### Output deviation / "loss" achieved (rollout, 1 sample, seed 0)

MSE is derived from PSNR at pixel range [0,1] (`MSE = 10^(-PSNR/10)`). The key figure is
**bf16-trim vs fp32 generated frames** (the only lossy step; A1/A3/A4 add nothing):

| Domain | comparison | PSNR (dB) | MSE | SSIM | LPIPS |
|---|---|---|---|---|---|
| pusht B/2 | bf16-trim vs fp32 gen | 52.9 | **5.1e-6** | 0.9976 | 0.0003 |
| csgo L/2 | bf16-trim vs fp32 gen | 44.8 | **3.3e-5** | 0.9889 | 0.0047 |
| pusht B/2 | gen vs GT (fp32→bf16) | 35.08 → 35.08 | 3.1e-4 | 0.9857 → 0.9856 | 0.0103 → 0.0100 |
| csgo L/2 | gen vs GT (fp32→bf16) | 16.53 → 16.50 | 2.2e-2 | 0.485 → 0.484 | 0.387 → 0.388 |

Interpretation: bf16 perturbs generated pixels by MSE ~5e-6 (pusht) / ~3e-5 (csgo) — visually
lossless. Gen-vs-GT quality is unchanged by the optimizations (csgo's low absolute SSIM is the
model's own difficulty on that domain, identical in fp32).

### Speed (pure DiT sampling loop, s per generated frame, gfx1151; seed 0)

| Variant | csgo L/2 | × | pusht B/2 | × |
|---|---|---|---|---|
| fp32 full-window (upstream-equiv) | 89.3 | 1.0× | 6.4 | 1.0× |
| fp32 trim (A1/A3/A4, exact) | 26.8 | 3.3× | 3.3 | 2.0× |
| bf16 full-window | 19.0 | 4.7× | 1.3 | 5.0× |
| **bf16 trim (A1/A3/A4/B1)** | **5.0** | **17.9×** | **0.7** | **9.0×** |

Orthogonal: exact trim ≈ 2–3.3×; bf16 ≈ 4.7–5×; stacked ≈ 9× (B/2) / 18× (L/2).

### Artifacts
- 4-way collages `[GT | fp32 | fp32-trim | bf16 | bf16-trim]`:
  `artifacts/nano-world-model/opt_val/collage_{csgo,pusht}_4way.mp4`
- 2-column GT|gen compares + metrics JSON in the same folder.

---

## Round 2 — A2 (temporal adaLN de-duplication)

**What.** In the temporal blocks the conditioning `timestep_temp` is `repeat(t_emb, 'b f d ->
(b p) f d', p=P)` — the same per-frame cond copied across all `P` spatial patches (P=256 for
L/2, 64 for B/2) — before the `Linear(D→6D)` adaLN projection. Since adaLN is linear,
`adaLN(repeat(c0)) == repeat(adaLN(c0))`, so `(P-1)/P` of that GEMM is redundant. A2 projects the
`B·F` unique rows once and broadcasts the 6 modulation tensors to `B·P·F`, and skips materializing
`timestep_temp` entirely. Gated (`model._a2` / `NANOWM_A2=0`); enabled only for the exercised safe
case (per-frame timesteps, no class/text extras, action fused into `x` not the cond).

**Exactness.** A2-on vs A2-off (same model/weights/input): max abs **3–5e-6** (fp reduction-order
noise). Full parity vs unmodified upstream still **PASS** (A3 3–6e-6, A1 2e-6, A4 bit-exact) — A2
does not perturb the exact-path guarantee.

**Speed (per-forward A/B, A2 off→on, gfx1151).** Consistent ~1.2× in fp32; marginal in bf16.

| Domain | window | fp32 off→on | fp32 × | bf16 off→on | bf16 × |
|---|---|---|---|---|---|
| pusht B/2 | trim (F=2) | 72.8→62.3 ms | 1.17× | 14.0→13.2 ms | 1.06× |
| pusht B/2 | full (F=4) | 138.2→117.2 ms | 1.18× | 25.4→23.6 ms | 1.08× |
| csgo L/2 | trim (F=5) | 617.5→515.5 ms | 1.20× | 99.7→99.3 ms | 1.00× |
| csgo L/2 | full (F=16) | 1957→1643 ms | 1.19× | 376→351 ms | 1.07× |

**Read.** A2 is exact and delivers a clean, reproducible **~1.2× in fp32** on both scales — the
adaLN projection GEMM is genuinely redundant. But it is **marginal in the bf16 production path**
(~1.0–1.08×): once bf16 makes that GEMM cheap, the temporal block's wall-clock is dominated by
attention + the elementwise modulate/gate ops (which run on the full `B·P·F` tensor and are *not*
removed by A2). So the static FLOP-share of adaLN overstated its bf16 wall-clock share. Verdict:
**keep default-on** — it is free, exact, never regresses, and meaningfully helps the fp32 (exact-
only, no-bf16) deployment; the bf16 rollout gains little. Not the dramatic L/2 win the FLOP count
implied. _Possible follow-up: apply the modulation in a grouped `[B,P,F,D]` view (broadcast, no
expand-materialize) to claw back the bf16 case — deferred as low-payoff._

## Round 3 — C1 (fewer sampling steps)

**What.** `--num_sampling_steps` maps straight to `create_diffusion(timestep_respacing=...)` — a
first-class respaced-DDIM knob, **no model change**. Sampling wall-clock is linear in steps
(loop = `steps × per-forward`), so cutting steps is a direct multiplier. The question is the
**quality knee**. Swept steps {50,32,25,20,16,12,8} on the bf16-trim production variant, fixed
seed, measuring each vs GT and vs the 50-step reference (per-forward from the A2 bench:
pusht bf16-trim ≈13 ms → 50 steps ≈0.66 s/frame; csgo ≈99 ms → 50 steps ≈5.0 s/frame).

_Note: per-run total wall-clock (~146 s pusht / ~190–205 s csgo) is dominated by model+VAE load,
not sampling; the step-count effect is only visible in csgo totals (205→185 s). Speed is taken
from the linear sampling model, not these totals._

**pusht B/2** (reference 50-step vs GT: PSNR 35.09, SSIM 0.9857):

| steps | vsGT PSNR | vsGT SSIM | vsGT LPIPS | vs-50 PSNR | vs-50 SSIM | vs-50 LPIPS |
|---|---|---|---|---|---|---|
| 50 | 35.09 | 0.9857 | — | — | — | — |
| 32 | 32.41 | 0.9777 | 0.018 | 35.15 | 0.978 | 0.014 |
| 25 | 33.55 | 0.9846 | 0.012 | 36.63 | 0.985 | 0.010 |
| 20 | 34.95 | 0.9871 | 0.009 | 39.29 | 0.989 | 0.006 |
| 16 | 33.47 | 0.9820 | 0.013 | 36.27 | 0.985 | 0.010 |
| 12 | 34.34 | 0.9823 | 0.012 | 36.26 | 0.983 | 0.010 |
| 8  | 33.88 | 0.9857 | 0.011 | 37.67 | 0.987 | 0.009 |

→ On the control/robotics domain, output is **essentially step-count-insensitive**: even 8 steps
stays within LPIPS ≤0.014 / SSIM ±0.005 of the 50-step result (differences are sample noise, not
monotone degradation). **~3–6× on the sampling loop for free.** Recommended default **16 steps**
(3.1×), aggressive **8** (6.25×).

**csgo L/2** (reference 50-step vs GT: PSNR 16.50, SSIM 0.4837):

| steps | vsGT PSNR | vsGT SSIM | vsGT LPIPS | vs-50 PSNR | vs-50 SSIM | vs-50 LPIPS |
|---|---|---|---|---|---|---|
| 50 | 16.50 | 0.484 | — | — | — | — |
| 32 | 18.04 | 0.550 | 0.351 | 21.76 | 0.614 | 0.313 |
| 25 | 18.13 | 0.577 | 0.281 | 19.19 | 0.576 | 0.336 |
| 20 | 20.32 | 0.667 | 0.217 | 18.97 | 0.561 | 0.345 |
| 16 | 19.58 | 0.619 | 0.274 | 19.88 | 0.595 | 0.322 |
| 12 | 18.43 | 0.582 | 0.300 | 19.39 | 0.588 | 0.333 |
| 8  | 18.82 | 0.583 | 0.348 | 20.32 | 0.609 | 0.331 |

→ csgo is a **high-entropy autoregressive video** domain: rollouts diverge chaotically, so
LPIPS-to-the-50-step-reference is large (~0.31–0.35) and **roughly constant** even at 32 steps —
per-pixel closeness to the 50-step run is *not* a meaningful target here. Crucially, **fewer steps
does not hurt frame fidelity vs GT** (SSIM/PSNR are flat-to-better at 16–25 steps; the 50-step run
is not a gold standard at SSIM 0.48). Single-sample, 6–9 gen frames → noisy; recommend a
conservative **25–32 steps** for csgo pending a larger-sample / FVD study.

**Verdict.** C1 is the highest-value remaining lever and it is precision-independent. Control
domains: **16 steps** (3.1× sampling) at no measurable quality cost; video-game domain: **25–32**
conservatively. DPM-Solver++ integration (to push csgo below 25) is deferred — it must be threaded
through the diffusion-forcing per-frame scheduling matrix and is only worth it if a larger csgo
study shows DDIM degrading below ~25 steps.

**Decision (dev):** keep `num_sampling_steps=50` as the shipped default (no behavior change), expose
it as a first-class tunable knob (`NUM_SAMPLING_STEPS` env → `--num_sampling_steps`; already wired).
Users trading speed for quality can drop to 16 (control) / 25–32 (csgo) per the curves above.

## Round 4 — A5 (context-frame KV cache)

**What.** The sliding-window rollout denoises the target frame over N steps while **the context
frames are frozen** — `dfot_sample_loop` restores them every step (`img = where(context_mask,
img_prev, img)`) and pins their timestep at `t_stab`. With **causal** temporal attention, each
context frame attends only to earlier context, so its full per-layer representation (and hence its
temporal K/V) is invariant across all N steps *and* independent of the target. A5 therefore:
- **primes** once per window (first call): a normal full-window forward that additionally stashes,
  per temporal block, the context K/V (post-QK-norm, post-RoPE);
- **uses** the cache on every subsequent step: runs only the **target frame** through spatial +
  temporal, prepending the cached context K/V (with a bottom-right-shifted causal mask — SDPA's
  `is_causal` mis-aligns for non-square q/kv) and offsetting temp-embed/RoPE to the target's
  absolute position. Context output slots are zero-filled (the DDIM loop discards them).

Prime/use is auto-detected inside `forward` by `torch.equal` on the (restored) context latents, so
`dfot_sample_loop` is untouched. Gated: causal models only, opt-in via `a5_configure()` /
rollout default-on (`--no_a5` disables).

**Exactness (per-forward, target frame vs the non-A5 full-window forward).**

| | fp32 (prime / use / reprime) | bf16 (use) |
|---|---|---|
| pusht B/2 | 0.0 / 3.2e-6 / 3.8e-6 | 1.6e-2 |
| csgo L/2 | 0.0 / 3.3e-6 / 4.8e-6 | 1.2e-2 |

fp32 is exact (fp reduction-order noise); prime is bit-exact. bf16 deviation is pure bf16
rounding, same magnitude as B1's own bf16-vs-fp32.

**Speed — per-step DiT forward (A5 off→on, steady state).** Scales ~`window/1`:

| Domain | window (n_ctx+1) | fp32 full→use | bf16 full→use |
|---|---|---|---|
| pusht B/2 | 2 (1+1) | 65.9→36.7 ms (**1.80×**) | 13.1→9.3 ms (**1.40×**) |
| csgo L/2 | 5 (4+1) | 561→136 ms (**4.12×**) | 95.7→35.1 ms (**2.72×**) |

**End-to-end (bf16-trim rollout, A5 on vs off, same seed).** Quality unchanged (A5on-vs-A5off:
pusht PSNR 53.0 / SSIM 0.9976 / LPIPS 3e-4 / MSE ~5e-6; csgo 45.8 / 0.9905 / 4e-3 / MSE ~3e-5 —
the bf16 floor), gen-vs-GT identical (35.10≈35.09; 16.48≈16.50). csgo total wall-clock 205.6→187.8 s
(the sampling loop is only a fraction of the load-dominated total; the clean gain is the 2.72×
per-step above).

**Verdict.** A5 is exact and, unlike A2, it cuts real attention/MLP work so it **helps bf16
strongly** (csgo 2.72× per DiT step) and **stacks with C1** (fewer steps × cheaper steps). Biggest
remaining exact win for multi-context-frame domains (csgo). **Default-on for causal models.**

## Round 5 — D1 (torch.compile)

**What.** Wrap the DiT forward in `torch.compile(model, dynamic=False)` for TorchInductor kernel
fusion (elementwise modulate/gate chains, adaLN, RoPE, LayerNorm) on the **A5-off static-shape
path** (fixed trimmed window `history+1`), which composes cleanly with A1/A2/B1. Bench = eager vs
compiled per-forward latency + compiled-vs-eager output parity, fp32 & bf16, both scales (gfx1151,
15 iters / 6 warmup, first compiled call excluded from timing).

**Parity (compiled vs eager, same weights/input).**

| Domain | fp32 max_abs | bf16 max_abs |
|---|---|---|
| pusht B/2 | 1.29e-5 | 6.25e-2 |
| csgo L/2 | 6.32e-6 | 3.12e-2 |

fp32 is exact (~1e-5 fusion reordering noise); bf16 is ordinary bf16 rounding (same band as B1).

**Speed (per-forward, eager→compiled).**

| Domain | window | fp32 eager→comp | fp32 × | bf16 eager→comp | bf16 × |
|---|---|---|---|---|---|
| pusht B/2 | 2 | 65.9→53.4 ms | 1.23× | 13.1→10.6 ms | 1.23× |
| csgo L/2 | 5 | 568.3→435.3 ms | 1.31× | 92.7→72.7 ms | 1.27× |

**Read.** A consistent, exact-in-fp32 **~1.23–1.31×** across every config — free fusion win, no
model change. It is **orthogonal to A1/A2/B1** (same static shapes) so it stacks with the whole
round-1/2 stack. It does **not** compose with A5: A5's prime/use modes have different frame counts
(dynamic shapes) and branch on `torch.equal(context)` (data-dependent control flow), which force
graph breaks / recompiles — and A5 alone already delivers a bigger per-step win (csgo 2.72× bf16)
than D1's 1.3×, so they are **alternatives, not a stack**. Cost: first-call compilation latency
(seconds), amortized over a rollout but a per-process warmup tax.

**Verdict.** Exact in fp32, stable on ROCm/Inductor (gfx1151, PyTorch 2.10), reproducible ~1.3×.
**Recommend opt-in** (not forced-on) for the A5-off deployment — the compile-warmup tax and the
mutual-exclusion with A5 mean it should be a flag, most valuable for the fp32 exact-only path and
the pusht/B-scale (single-context) case where A5's win is smaller. For multi-context csgo, prefer
A5. Deferred: A5+compile via `torch._dynamo` region markers to compile only the (static) per-block
compute inside A5's use-path — low payoff vs. effort.

---

## Stacking summary (what ships)

- **Default-on, exact:** A1 (trim) + A3 (RoPE) + A4 (action cache) + A2 (adaLN dedup) + A5 (KV
  cache, causal only). B1 (bf16) default-on for the production path (fp32 exact path preserved).
- **Tunable:** C1 sampling steps (`NUM_SAMPLING_STEPS`, default 50; 16 control / 25–32 csgo).
- **Opt-in:** D1 `torch.compile` (A5-off path only; best for fp32 / single-context).

---

## Full-stack composite — cumulative waterfall (measured)

One apples-to-apples per-step DiT-forward bench that stacks each optimization on top of the previous
(`patches/opt/stack_bench.py`, gfx1151, 15 iters/6 warmup). Each layer adds exactly one optimization;
A3 is baked into the optimized forward and A4 is a per-step loop amortization (negligible per-forward,
so folded into the trimmed layers). D1 is excluded — it does not compose with A5 (see round 5) and A5
wins. bf16 is the production path; the fp32 layers are the exact-only reference.

**pusht B/2** (full window F=4 → trim F=2, n_ctx=1):

| Layer | per-step DiT | cumulative × (vs V0) | marginal × |
|---|---|---|---|
| V0 baseline (fp32, full window) | 137.4 ms | 1.00× | — |
| +A1 trim (fp32, exact) | 72.6 ms | 1.89× | 1.89× |
| +A2 adaLN dedup (fp32, exact) | 62.1 ms | 2.21× | 1.17× |
| +B1 bf16 | 13.3 ms | 10.32× | 4.67× |
| +A5 KV cache (bf16, use) | 9.3 ms | **14.78×** | 1.43× |

**csgo L/2** (full window F=16 → trim F=5, n_ctx=4):

| Layer | per-step DiT | cumulative × (vs V0) | marginal × |
|---|---|---|---|
| V0 baseline (fp32, full window) | 1954.2 ms | 1.00× | — |
| +A1 trim (fp32, exact) | 613.7 ms | 3.18× | 3.18× |
| +A2 adaLN dedup (fp32, exact) | 516.0 ms | 3.79× | 1.19× |
| +B1 bf16 | 97.0 ms | 20.14× | 5.32× |
| +A5 KV cache (bf16, use) | 35.0 ms | **55.90×** | 2.78× |

(A5 prime forward — paid once per window — is 13.3 ms pusht / 96.9 ms csgo, i.e. one full-window-
equivalent step; every subsequent step is the cheap "use" cost above.)

### Per-generated-frame (full sampling loop = one prime + (S−1) uses)

| Domain | V0 baseline @50 | full stack @50 | × | full stack + C1 | × |
|---|---|---|---|---|---|
| pusht B/2 | 6.87 s | 0.469 s | **14.6×** | 0.153 s @16 steps | **45.0×** |
| csgo L/2 | 97.71 s | 1.810 s | **54.0×** | 0.936 s @25 steps | **104.4×** |

### Full-stack exactness (target frame vs the V0 upstream forward)

| | +A1 fp32 | +A2 fp32 | +A5 fp32 (full exact stack) | bf16 stack vs fp32 (only lossy step) |
|---|---|---|---|---|
| pusht B/2 | 2.38e-6 | 2.38e-6 | 3.34e-6 | 2.58e-2 |
| csgo L/2 | 3.87e-6 | 3.10e-6 | 3.28e-6 | 1.88e-2 |

The entire **exact** stack (A1+A2+A3+A4+A5, fp32) reproduces the upstream target to ~3e-6 (fp
reduction-order noise). The **only** lossy step is B1's bf16 rounding (~2e-2 per-forward), whose
end-to-end pixel effect was measured as visually lossless (round 1: bf16-trim-vs-fp32 gen MSE
5.1e-6 pusht / 3.3e-5 csgo).

### End-to-end quality (the full stack rolled out)

The round-4 A5-on rollout **is** the full stack run end-to-end (A2/A1/A3/A4/B1 are all default-on
when A5 is enabled). Gen-vs-GT is unchanged from the fp32 upstream: pusht PSNR 35.10 (vs 35.09
fp32), csgo 16.48 (vs 16.50 fp32); full-stack-vs-fp32-gen PSNR 53.0/45.8, SSIM 0.9976/0.9905,
LPIPS 3e-4/4e-3. **Stacking everything is exact-to-bf16-noise with no quality regression.**

**Overall outcome.** The complete shipped stack delivers a measured **14.6× (pusht) / 54.0× (csgo)**
speedup per generated frame at unchanged 50 steps and unchanged output quality, rising to
**45× / 104×** when combined with the C1 step reduction — dominated by B1 (bf16, ~4.7–5.3×) and A5
(KV cache, up to 2.78× on the deeper multi-context csgo), with A1/A2 contributing the exact
front-end (1.9–3.8×) and C1 the precision-independent 3.1× (pusht) / 2× (csgo) tail.
