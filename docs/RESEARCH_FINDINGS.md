# FastWAM — Research Findings Log

Persistent, append-only log of short findings from profiling/optimizing FastWAM on AMD
Strix Halo (`gfx1151`, ROCm, bf16). Newest on top. Keep entries terse; link artifacts
under `artifacts/` and raw metrics under `tmp/bench/` (laptop-only per workspace rule 3).

---

## 2026-07-06 — Planning-path exact-caching optimizations (measured 1.43x)

**Context.** Full model breakdown of the deployed RoboTwin checkpoint
(`robotwin_uncond_3cam_384.pt`), planning path `infer_action`. Artifacts:
`artifacts/fastwam_breakdown/` (plots 1-6, `SUMMARY.md`), raw
`tmp/bench/metrics.json` + `tmp/bench/metrics_opt.json`.

**Architecture.** Wan2.2 world-action Mixture-of-Transformers, 12.4 B params total:
- Text encoder UMT5-XXL **5.68 B**; VAE (Wan2.2) **0.70 B**; MoT video expert **5.0 B**;
  MoT action expert **1.02 B**. MoT DiT = 30 layers, 24 heads x 128 = hidden 3072, ffn 14336.
- Deployed dims: action_dim 14, action_horizon 32, replan every 24 actions,
  num_inference_steps 10, VAE spatial /16 temporal /4 (z_dim 48).
- Captured tokens: text 128, first-frame latent [1,48,1,24,20], video tokens 120 (1 frame).

**Planning vs full video (measured, gfx1151 bf16, 10 steps).**
- Planning `infer_action` ~852 ms/plan: text enc 154 (17%), VAE enc 83 (9%),
  video prefill 173 (19%), action denoise loop 46 ms x10 = 468 (52%). **No VAE decode.**
  KV-cache design: video branch prefilled ONCE into per-layer K/V; each step only runs the
  small action expert against cached video K/V.
- Full video `infer_joint` ~5.5 s: joint MoT steps 306 ms x10 (56%) + **VAE decode 2.2 s (40%)**.
  ~6.2x planning; the deployed policy avoids all of it.
- Latency is memory/launch-bound at these tiny seqs (action step ~9 TFLOP/s effective), not
  FLOP-bound; the FFN + linear projections dominate, attention itself is negligible.

**Two exact optimizations (bit-identical output, max |Δaction| = 0.0):**
1. **Text-encode cache across an episode** — UMT5-XXL (154 ms) re-runs every replan although
   the instruction is constant per episode. Memoize `encode_prompt` by prompt string.
   → 852 → 699 ms (**1.22x**).
2. **Text cross-attn K/V + text-embedding MLP cache across denoise steps** — within one plan,
   context (text+proprio) is constant across the ~10 diffusion steps, yet each step recomputes
   `action_expert.text_embedding(context)` and every layer's cross-attn `k(ctx)/v(ctx)`.
   Cache them once per plan. → step 46.0 → 35.6 ms; total 699 → 593 ms (**1.43x combined**).

**Provenance.** Model code is upstream-pristine (`git status --porcelain` empty; HEAD 45d8e14,
PR #20). These are optimization *gaps* in the author's code (they cache only video self-attn
K/V), **not** porting regressions. Both are inference-only reuse of provably-constant tensors →
exact, upstream-contributable, zero quality cost.

**Re-download note.** DiffSynth downloads to `./checkpoints` (ephemeral) unless
`DIFFSYNTH_MODEL_BASE_PATH` is set. Project already sets it correctly
(`packages/wam/fastwam/config.yaml:21` → `/models/diffsynth`); host cache lives at
`workspace/fastwam/models/diffsynth`. The re-download only affected an ad-hoc benchmark launcher.

**Attention backend (measured, gfx1151 bf16).** `flash_attention()` is
`F.scaled_dot_product_attention` ("compatibility_mode"), always called with an `attn_mask`.
Per-call SDPA latency for the model's shapes (ms):

| shape | default | MATH | EFFICIENT | FLASH |
|---|---|---|---|---|
| action self mixed (q32,kv152, masked) | 0.054 | 0.112 | 0.053 | unsupported (mask) |
| action cross-text (q32,kv129, no mask) | 0.051 | 0.095 | 0.041 | 0.036 |
| video prefill self (q120,kv120, masked) | 0.054 | 0.231 | 0.058 | unsupported (mask) |
| joint self (q392,kv392, masked) | 0.288 | 1.245 | 0.284 | unsupported (mask) |

Findings: torch's **default already dispatches to the memory-efficient backend** (≈EFFICIENT,
~4x faster than MATH), so we are NOT stuck on the slow math path. **AOTriton FLASH is unavailable
whenever an `attn_mask` is passed** on gfx1151 ("No available kernel"); it works only for the
maskless cross-attn and is then marginally faster (0.036 vs 0.041 ms) — negligible. Attention is
a tiny fraction of step time (~30 layers x ~0.05 ms ≈ 1.5 ms vs a 35 ms step); the step is
FFN/projection-bound (launch/memory-bound), not attention-bound. **Conclusion: no meaningful win
from switching attention backends; the caching is the right lever.**

**Status: LANDED.** In-source patch `packages/wam/fastwam/patches/fastwam_kv_cache.patch`
(4 files, applies clean on 45d8e14, `git apply --check` OK). Re-validated from source:
max |Δaction| = 0.0, 852.8 → 589.4 ms (**1.447x**). Baked into the FastWAM image (Dockerfile
`git apply` after checkout) and default-on for every consumer (open-/closed-loop + interactive,
LIBERO + RoboTwin); kill-switch `FASTWAM_TEXT_KV_CACHE=0` wired in `config.yaml`.
Future: extend the text cross-attn K/V cache to `infer_joint` (full-video) too.

**Image rebuilds (2026-07-06).**
- `fastwam-robotwin` (on `sim-robotwin`): rebuilt OK, patch verified in image; baked-image
  validation via the real `deploy_policy` path with default env (`FASTWAM_TEXT_KV_CACHE` unset →
  on): max |Δaction| = 0.0, 883.5 → 611.3 ms (**1.445x**). RoboTwin closed-loop + interactive
  now cache by default.
- `fastwam-libero` (on `sim-libero`): rebuilt OK, patch verified in image.
- `fastwam` (plain, on `ryzer_env`/rocm base): **build blocked by a PRE-EXISTING issue unrelated
  to caching** — upstream `pyproject` pins `numpy==1.26.4` while the plain base ships `numpy 2.4.4`
  → `ResolutionImpossible`. Sim bases already use numpy 1.26.4 so they are unaffected.
  `strip_cuda_torch.py` intentionally strips only the torch stack (keeps upstream pins faithful),
  so it does not strip numpy. The caching patch IS wired into this Dockerfile and will apply once
  the numpy mismatch is resolved (options: also strip numpy + validate FastWAM on numpy 2.x, or
  rebuild the plain layer on a numpy-1.26 base). Deferred — needs a decision, not a silent change.

**Correctness of closed-loop:** since actions are bit-identical (Δ=0) through the full deploy
path on real image+proprio inputs, closed-loop task success is provably unchanged; a full rollout
is optional confirmation only.
