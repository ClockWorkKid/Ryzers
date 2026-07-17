# NanoWM optimization scoping (Strix Halo / gfx1151)

Scoping pass to find redundant compute, cache-reuse, and ROCm-native acceleration opportunities in
the released-checkpoint inference path, ranked by ROI. Grounded in the measured forward-pass
analysis (`FORWARD_PASS_ANALYSIS.md`) and a code read of the upstream compute path
(`src/models/nanowm.py`, `src/diffusion/{gaussian_diffusion,df_sample}.py`, `src/sample/rollout.py`).
No optimizations implemented yet — this is the map before the run.

## How inference actually runs (measured + code)

Sliding-window diffusion-forcing rollout (`rollout.py` → `dfot_sample` → `dfot_sample_loop`):

- **One output frame per window.** `n_generate_frames=1`, `scheduling_mode="sequential"` — each output
  frame is a full 50-step DDIM denoise, then the window slides by 1 and repeats. A CSGO sample
  (~46 gen frames) is ~46×50 ≈ 2300 DiT forwards; the DiT sampling loop is >98% of wall-clock.
- **No CFG in the rollout path** (`dfot_ddim_sample` calls plain `model(x, t)`), so `forward_with_cfg`
  and its 2× cost are not in play here.
- **Window = `model.num_frames` (16 for CSGO, 4 for B/2)**, but `history_length` = `n_context_frames`
  (4 for CSGO, 1 for B/2). With sequential scheduling + `n_generate_frames=1`, the frame being
  denoised sits at index `history_length`; **all frames after it are pure noise and are masked out by
  causal temporal attention** — computed every step, then discarded.
- VAE runs once (encode context, decode all at the end) — ~1.3 s/sample, negligible. Not a target.

## Bottleneck recap (from measurements)

- fp32 DiT forward: **B/2 131–164 ms, L/2 3028–3230 ms.**
- L/2 fp32 latency split: **adaLN cond 1457 ms (45%)**, MLP 999 ms (31%), attention 659 ms (20%).
- bf16/fp16 autocast: **B/2 5.7×, L/2 8.1×** faster (fp32 GEMMs fall back to slow kernels).

---

## Opportunities (ranked by ROI)

### A. Exact / lossless (no change to outputs)

**A1. Causal window truncation — generate only `history_length + 1` frames, not `model.num_frames`.**
`rollout.py:103` sets `num_frames_per_inference = args.model.num_frames`. With sequential +
`n_generate_frames=1` and **causal** temporal attention (`nanowm.py:645`, `is_causal=self.causal`,
configs set `causal: true`), the target frame at index `history_length` only attends to frames
`0..history_length`. Trailing frames contribute nothing.
- CSGO: 16 → 5 frames (**~3.2× fewer spatial-block rows and temporal work**). B/2: 4 → 2 (~2×).
- Exactness: bit-identical for the kept frame (positions 0..h unchanged; causal mask already drops >h).
- Change: `rollout.py` set window = `history_length + 1`; model must accept `F < num_frames` — the
  only hard dependency is `nanowm.py:621` `x = x + self.temp_embed` → slice `self.temp_embed[:, :F]`
  (RoPE at `:84` already slices `freqs[:N]`). ~3 lines. **Effort: low. Risk: low.**

**A2. De-duplicate the temporal adaLN projection (attacks the #1 L/2 cost).**
For temporal blocks the conditioning is `timestep_temp = repeat(t_emb, 'b f d -> (b p) f d', p=256)`
(`nanowm.py:577`) — **256 identical copies across patches** — and each block then runs
`adaLN_modulation(c)` = `Linear(D→6D)` on all `B·P·F` rows (`nanowm.py:313`). The 6 shift/scale/gate
tensors are identical across the 256 patch copies, so 255/256 of that projection is redundant.
- Fix: compute `adaLN_modulation` on the unique `[B, F, D]` (P× fewer rows) and broadcast the 6
  params when applying `modulate`/gate. Same for `action_adaLN_modulation` (adaln injection) — though
  released checkpoints use `additive`, where `action_emb` is just added to `x` and the temporal repeat
  (`nanowm.py:561`) can likewise be deferred/broadcast.
- Impact: the temporal adaLN GEMM shrinks ~256×; adaLN is 45% of L/2 fp32 latency (and a big chunk of
  its 1457 ms is the `m≈6144` fallback GEMM). Compounds with bf16.
- Change: refactor `TransformerBlock.forward` temporal branch + `NanoWM.forward` to pass unique cond.
  **Effort: medium. Risk: low–medium (needs careful broadcast + numeric parity check).**

**A3. Precompute RoPE cos/sin once.** `apply_rotary_pos_embed` (`nanowm.py:78–88`) recomputes
`freqs.cos()/.sin()` and `freqs.to(x.device)` on **every** attention call (12–24 blocks × 50 steps ×
frames). freqs are constant buffers. Precompute `cos`/`sin` buffers at init (spatial + temporal).
- Impact: small FLOPs but removes thousands of tiny kernels/launches per sample (launch-bound on iGPU).
- **Effort: low. Risk: none (exact).**

**A4. Cache action embedding across the 50 steps of a window.** `action_emb` depends only on `action`
(`nanowm.py:541`), which is constant across a window's sampling steps, yet it is recomputed each step.
- Impact: small FLOPs; removes the action MLP + shift + two `repeat`s per step.
- Change: hoist action embedding out of the per-step forward (compute once per window, pass in), or
  memoize on an unchanged-action guard. **Effort: low–medium. Risk: low (exact).**

**A5. Context-frame KV cache across sampling steps.** Context frames (`0..history_length−1`) are clean
and fixed across all 50 steps of a window; after A1 they are `history_length` of `history_length+1`
frames. Their per-block spatial outputs and temporal K/V are constant across steps → cache and reuse,
denoising only the single target frame's query path.
- Impact: potentially large for CSGO (4 of 5 frames become cache hits after A1), but overlaps A1.
- Change: per-block K/V cache + restructure `forward` to accept cached context. **Effort: high.
  Risk: medium. Exact if done right.** Defer until A1/A2 land and are measured.

### B. Precision / ROCm-native kernels (measured wins; numerical, validate quality)

**B1. bf16 autocast for the DiT sampling loop — highest ROI, already measured.** Wrap the per-step
`model(x,t)` in `torch.autocast("cuda", dtype=bfloat16)`. **5.7× (B/2), 8.1× (L/2)** on the forward.
Keep VAE in fp32 (cheap, avoids decode artifacts). bf16 preferred over fp16 for diffusion stability.
- This is what makes flash/aotriton attention engage: SDPA flash errors under fp32 on ROCm but runs
  in bf16/fp16 (knob sweep). So B1 delivers ROCm-native flash attention "for free".
- **Effort: low. Risk: low–medium — must validate rollout quality vs fp32 (SSIM/LPIPS/FVD).**

**B2. Keep `TORCH_BLAS_PREFER_HIPBLASLT=0`.** A/B confirmed `=1` is slower for fp32 (B/2 164→205 ms,
L/2 3028→4885 ms). Re-A/B after bf16 lands, since the fused hipBLASLt path may win in low precision.

**B3. Leave SDPA on `default`.** `default`≈`flash`; forcing `math` is ~15–20% slower. No action beyond
ensuring nothing forces the math backend.

### C. Sampler / algorithmic (fewer forwards; validate quality)

**C1. Fewer sampling steps / faster sampler.** 50 DDIM steps scale cost linearly. 20–25 steps or a
DPM-Solver++/DDIM-eta schedule roughly halves forwards. Orthogonal to A/B, compounds multiplicatively.
- **Effort: low (steps) / medium (new sampler). Risk: medium — quality check required.**

### D. Compilation / fusion (opportunistic)

**D1. `torch.compile` the DiT forward** (there is a commented `@torch.compile` at `nanowm.py:512`).
Inductor can fuse the adaLN `chunk(6)` + `modulate` + gated residual elementwise chains and the
RMSNorm/RoPE ops. ROCm inductor stability is the risk. **Effort: medium. Risk: medium. ~1.2–1.5×.**
**D2. Hand-fuse adaLN `chunk`/`modulate`/gate** if compile is unstable. **Effort: medium.**

---

## Recommended sequence and projected CSGO impact

Ordered to bank the safe, measured wins first, validating quality after each numerical step. Baseline
CSGO ≈ 2740 s/sample (fp32, 16-frame window).

| Step | Type | Est. cumulative CSGO s/sample | Notes |
|---|---|---|---|
| baseline | — | ~2740 | measured |
| B1 bf16 autocast | numerical | ~340 | measured 8.1× on forward; validate quality |
| + A1 window truncation 16→5 | exact | ~130–180 | ~2–3× on transformer (partial overlap w/ attn scaling) |
| + A2 temporal adaLN dedup | exact | ~110–150 | trims residual adaLN rows/bandwidth |
| + A3/A4 RoPE+action caching | exact | ~100–140 | launch-overhead reduction (iGPU is launch-sensitive) |
| + C1 steps 50→25 | numerical | ~55–75 | quality check required |
| + D1 torch.compile | fusion | ~45–65 | if ROCm inductor stable |

Net target ~**≥20× end-to-end** on CSGO (and similar structure, smaller multiplier, on B/2). Exact
steps (A1–A5) carry no quality risk; numerical steps (B1, C1) must be validated against the fp32
reference rollouts with SSIM/LPIPS (and FVD where feasible) before adoption.

---

## Round 1 — implemented + validated (A1 + A3 + A4 + B1)

Implemented as an inference-only override (`packages/vidgen/nano-world-model/patches/opt/`): `nanowm.py`
(A1 temp_embed slice, A3 precomputed RoPE cos/sin, A4 precomputed action_emb kwarg) and `rollout.py`
(A1 causal window trim gated on `model.causal`, A4 embed-once-per-window, B1 `--dit_autocast bf16`,
`--seed`, `--full_window`). All optimizations default OFF / full-window so the upstream fp32 path is
byte-for-byte preserved.

### Module parity (modified vs UNMODIFIED upstream, identical weights)

| Check | pusht B/2 (max abs) | csgo L/2 (max abs) |
|---|---|---|
| A3 modified full-window fp32 vs original | 3.0e-6 | 5.9e-6 |
| A1 trimmed kept-frame vs original[:,h]   | 2.1e-6 | 3.7e-6 |
| A4 precomputed action_emb vs in-forward  | **0.0 (bit-exact)** | **0.0 (bit-exact)** |
| B1 bf16 vs fp32 (per forward, rel err)   | 2.3e-3 | 1.1e-2 |

A1/A3/A4 are numerically exact (only fp reduction-order noise). Exact-path parity **PASS** both variants.

### Rollout quality (1 sample, seed 0; gen vs GT, and bf16 vs fp32 on identical noise)

| Domain | run | PSNR | SSIM | LPIPS |
|---|---|---|---|---|
| pusht | fp32trim gen-vs-GT | 35.08 | 0.9857 | 0.0103 |
| pusht | bf16trim gen-vs-GT | 35.08 | 0.9856 | 0.0100 |
| pusht | **bf16 vs fp32 gen** | 52.9 | **0.9976** | 0.0003 |
| csgo | fp32trim gen-vs-GT | 16.53 | 0.485 | 0.387 |
| csgo | bf16trim gen-vs-GT | 16.50 | 0.484 | 0.388 |
| csgo | **bf16 vs fp32 gen** | 44.8 | **0.9889** | 0.0047 |

bf16 changes the generated frames negligibly (SSIM 0.99+ vs fp32) and does not move gen-vs-GT quality
(csgo's low absolute SSIM is the model's inherent difficulty on that domain, identical in fp32).
Compare videos: `artifacts/nano-world-model/opt_val/{csgo,pusht}_{fp32trim,bf16trim}_compare.mp4`.

### Speed (pure sampling loop, s per generated frame, gfx1151)

Full 2×4 matrix (single aligned run, seed 0; csgo 6 gen frames, pusht 7 gen frames):

| Variant | csgo L/2 s/frame | (× vs fp32full) | pusht B/2 s/frame | (× vs fp32full) |
|---|---|---|---|---|
| fp32 full-window | 89.3 | 1.0× | 6.4 | 1.0× |
| fp32 trim (A1/A3/A4, exact) | 26.8 | 3.3× | 3.3 | 2.0× |
| bf16 full-window | 19.0 | 4.7× | 1.3 | 5.0× |
| **bf16 trim (+B1)** | **5.0** | **17.9×** | **0.7** | **9.0×** |

Clean orthogonal decomposition: exact window-trim ≈ 2× (pusht) / 3.3× (csgo) with zero quality
change; bf16 ≈ 5× (pusht) / 4.7× (csgo); combined ≈ 9× / 18×.

### Side-by-side collage
`artifacts/nano-world-model/opt_val/collage_{csgo,pusht}_4way.mp4` — panels
`[ GT | fp32 | fp32-trim | bf16 | bf16-trim ]`, temporally aligned, seed 0.

### Not yet done (next rounds, from the ranked list)
A2 (temporal adaLN de-dup), A5 (context KV cache), C1 (fewer steps / faster sampler),
D1 (`torch.compile`). A2 is the highest-value remaining exact win for L/2.

## Constraints / conventions

- Per workspace rules: prefer upstream code, apply patches only for ROCm/project needs. A1/A2/A3 are
  genuine upstream inefficiencies — implement behind flags, keep fp32/full-window parity paths, and
  consider upstreaming. Validate each module independently on real data before assembling
  (per rule 2). Weights/large videos stay on the remote; only small artifacts + code mirror locally.
- Every numerical change must ship with a quality-parity artifact (GT-left / pred-right compare video
  + SSIM/LPIPS numbers) before it replaces the fp32 default.
