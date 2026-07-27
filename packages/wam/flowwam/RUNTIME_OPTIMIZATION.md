# FlowWAM — runtime optimization (P8)

Inference-time latency analysis + optimization of the FlowWAM closed-loop flow-action path on
**AMD Strix Halo `gfx1151`, ROCm 7.2.2, bf16** (image `flowwam-robotwin`). Companion to the
cross-model `docs/RUNTIME_OPTIMIZATION_PLAYBOOK.md`; section refs (§) point there. All numbers are
measured against the real loaded weights, isolated from the sim/websocket (rule 2).

## TL;DR
- **Bottleneck class: video-DiT-bound** — the *opposite* of FastWAM. FlowWAM fully denoises a
  dual-stream (RGB+flow) video every replan, so the **video DiT forward is 91.6% of a replan**; the
  action expert (the thing that outputs actions) is only 6.9%.
- **bf16 is already the standard route** for every sub-model (DiT/VAE/T5/FlowStream/IDM).
- **Two structural wins already in place**: the deployed path **never runs the VAE decoder** (§3.2)
  and the **T5 text encoder is episode-cached** (§2.2, per-instruction `_text_cache`).
- **Net quality-preserving speedup ≈ 1.07×** (cross-attn K/V cache 1.019× bit-exact + torch.compile
  1.051× near-lossless). The only *big* lever found — dual-stream asymmetry (1.61×) — is too lossy
  at DS=2 (task success 100%→60%) and needs a milder setting or a finetune to be viable.

## bf16 baseline (deployed control path, decoder off, text cached)
Warm steady-state replan = **80.97 s** (n=3, very stable; video 25 steps / action 50 steps).

| stage | calls | % replan |
|---|---|---|
| **video DiT dual-stream denoise** | 25 | **88.1%** (2845 ms/step) |
| capture per-layer feats | 1 | 3.5% |
| action-expert (IDM) denoise | 50 | 6.9% (111 ms/step) |
| VAE encode (rgb+flow) | 2 | 0.2% |
| scheduler/glue | — | 1.4% |

→ DiT forward (video + capture) = **91.6%** of the replan.

## Inside one DiT forward
Geometry: 30 blocks, dim 3072, 24 heads, head_dim 128, **ffn 14336**, 512 text tokens; latent
T=13, H=24, W=20 → ~1560 tokens/stream, **joint seq ~3120**.

| op | % forward | note |
|---|---|---|
| fp32 norms / modulate / heads / patchify (remainder) | ~36% | torch.compile target |
| **FFN GEMMs** (ffn=14336, M≈1560) | ~26% | big-GEMM |
| self-attn q/k/v/o proj | ~9% | |
| text cross-attn (512 tok) | ~8% | §2.1 K/V cache target |
| SDPA self-attn core | ~6% | **already flash/efficient (6 ms vs 90 ms math) — no lever** |
| RoPE (fp64) | ~3% | |

Ruled out by measurement: SDPA backend is already optimal; the DiT stays GPU-resident
(`cast_to` calls = 0 on the deployed path → no per-forward weight re-cast).

## Levers tested (25-step DiT-loop A/B vs bf16 baseline)

| Lever | Kind | Speedup | Quality | Verdict |
|---|---|---|---|---|
| Cross-attn K/V + text-embed cache (§2.1) | general | **1.019×** | bit-exact (max\|Δ\|=0) | **adopt** (free, minor) |
| torch.compile block fn (§4) | conditional | **1.051×** | near-lossless (cos 0.9999), one-time 78 s compile | **adopt** (modest) |
| W8A8 int8 FFN(+qkvo) (§2.5) | general* | **0.79×** | lossy | **REJECT — regression** |
| **Dual-stream asymmetry** (flow grid, unique) | model-specific | **1.27× @ 18×16 … 1.65× @ DS2** | cos ~0.93 (NOT near-lossless); closed-loop 18×16 preserves success (click_bell 9/10 vs 8/10 @1.28×), DS2 degrades -20 pts | **opt-in "fast" preset, NOT default** |
| fp64→fp32 RoPE | conditional | ~1.02× (est.) | max\|Δ\|=0.18, not exact | skip (low ROI) |

### Why W8A8 is a regression here (correction to the playbook hypothesis)
FlowWAM's hot loop *is* big-GEMM (ffn=14336, run 26×/replan), which suggested §2.5 quant should
pay off — but it does **not** on this ROCm stack. Isolated `torch._int_mm` at M=1560:
`ffn_up 3072→14336 = 0.28×`, `ffn_down 14336→3072 = 1.17×`, `qkvo 3072→3072 = 0.57×`. The per-token
dequant over the 14336-wide output is bandwidth-bound and `_int_mm` isn't tuned for the up-proj
shape. **Big-GEMM ≠ quant win on gfx1151.**

### FlowWAM-unique lever: dual-stream asymmetry (milder-grid sweep)
The flow stream shares the DiT blocks and is a secondary motion channel. Running it at reduced
spatial resolution shrinks the joint self-attention and the flow-side qkvo/FFN/RoPE — **no model_fn
change needed** (the flow grid/RoPE/head derive from the flow-latent shape). The full-res flow grid
matches RGB (**24×20 = 480 tok/frame**); DS=2 == 12×10. Sweep vs the full-res flow (25-step loop):

| flow grid | tok/fr | area% | speedup | rgb cos |
|---|---|---|---|---|
| 24×20 (full) | 480 | 100 | 1.00× | 1.0000 |
| 22×18 | 396 | 82.5 | 1.12× | 0.9307 |
| 20×18 | 360 | 75.0 | 1.14× | 0.9353 |
| **18×16** | 288 | 60.0 | **1.27×** | **0.9355** |
| 16×14 | 224 | 46.7 | 1.42× | 0.9099 |
| 14×12 | 168 | 35.0 | 1.51× | 0.9222 |
| 12×10 (=DS2) | 120 | 25.0 | 1.65× | 0.9212 |

**Key finding: RGB fidelity cliffs to cos ~0.93 at the *first* reduction and plateaus** — there is
**no near-lossless milder region**. Best speed-per-fidelity is **18×16 (1.27×, cos 0.9355)**.
Closed-loop on **discriminating** in-distribution tasks (10 ep/config, seed base 100000, identical
protocol). `beat_block_hammer` sits at the success ceiling (5/5 across all grids) and cannot separate
the configs, so we benchmarked in-distribution tasks whose baseline is *partial*:

| task | baseline 24×20 | 18×16 | DS2 12×10 |
|---|---|---|---|
| `click_bell` (10 ep) | 8/10 = 80% (1891 s) | **9/10 = 90% (1472 s, 1.28×)** | 6/10 = 60% (2271 s, 0.83×) |
| `lift_pot` (matched first-6 seeds) | 4/6 = 66.7% | **4/6 = 66.7% (identical)** | 3/6 = 50% |

**18×16 preserves task success** (click_bell ≥ baseline within episode noise; lift_pot bit-for-bit
matched on shared seeds) **while delivering a measured 1.28× closed-loop wall speedup** on click_bell.
**DS2 (12×10) is rejected**: -20 pts (click_bell) / -17 pts (lift_pot matched) AND *no* wall win —
the extra failures run to the full sim horizon, so the per-step inference saving is erased (2271 s >
1891 s baseline). `place_object_basket` and `handover_block` floor at ~0 and run ~30 min/ep (not
tractable / non-discriminating), so they are excluded.

**Verdict:** shippable **opt-in "fast" preset** — recommended `FLOW_GRID=18x16` (+1.27× on the
video-DiT stage; ~1.28× measured end-to-end closed-loop, composes with the baked cache+compile),
now **validated quality-preserving on discriminating tasks** but **still not a default** (cos ~0.93
is a real fidelity trade, and DS2 shows how the cliff degrades harder tasks with no wall win).
Server-side, env-gated via `FLOW_GRID` / `FLOW_DS` (`agent_scripts/flowwam_p8_patch_server2.sh` +
`flowwam_asym_clsweep.sh` / multi-task bench `flowwam_asym_bench.sh`; speed/fidelity sweep
`flowwam_p8_asym_sweep.py`).

## Baked default route (shipped)
The two quality-preserving levers are **baked into the default inference path** of the closed-loop
demo (`scripts/flowwam_opt.py` + `scripts/opt_launch.py` + `scripts/opt_python.sh`). The demo sets
`PYTHON=/ryzers/scripts/opt_python.sh`, which the upstream `start_server.sh` honors to route the
server through `opt_launch.py` → `flowwam_opt.patch_class()` **before** running the unedited
`flow_action_server.py` (rules 2.1 / 0.0 — no upstream edits). `patch_class()` wraps the shared
`model_fn_wan_video_dual_stream` and, on the first forward, installs the instance-level cache and
arms `torch.compile` on the built `dit`.

Stacked measurement (same harness, shipped `flowwam_opt.patch_class()`, real weights, 25-step loop):

| config | speedup vs eager | quality |
|---|---|---|
| cache only (bit-exact) | 1.019× | max\|Δ\|=0 |
| torch.compile only | 1.030× | near-lossless (rgb cos 0.99988) |
| **cache × compile (default)** | **1.042×** | near-lossless (rgb cos 0.99995, flow 0.99997) |

The two levers **compose** (1.042× > either alone): the cache stays eager (`torch._dynamo.disable`)
so `torch.compile` fuses the norm/modulate/FFN tail while the cross-attn K/V is served from cache.
All deviation is bf16 rounding from `compile` (the cache is bit-exact). Closed-loop re-validated on
RoboTwin `beat_block_hammer` (see `agent_scripts/flowwam_opt_clval.sh`).

Env kill-switches (all default ON): `FLOWWAM_OPT=0` (all), `FLOWWAM_CACHE=0`, `FLOWWAM_COMPILE=0`,
`FLOWWAM_COMPILE_MODE=<inductor mode>`. Any failure falls back to plain eager, so a demo can never
break because of an optimization. First replan pays a one-time ~80 s Inductor compile.

## Net + open levers
- **Shipped (default, quality-preserving) = 1.042×**: cross-attn K/V + text cache (bit-exact) ×
  torch.compile[block] (near-lossless), baked into the closed-loop demo route.
- **Opt-in fast preset = +1.27× (18×16 flow)**: dual-stream asymmetry sweet-spot (composes with the
  default route → ~1.3×+ on the video-DiT stage). **Validated quality-preserving on discriminating
  closed-loop tasks** (click_bell 9/10 vs 8/10 baseline @1.28× wall; lift_pot matched) → env knob
  `FLOW_GRID=18x16`, **not default** (cos ~0.93; DS2 12×10 degrades -20 pts with no wall win).
- **Open**: fewer NFE / distillation (§2.6, out of P8 scope — the known knob); wider multi-seed
  sweep of 18×16 only if it is ever promoted toward default.

## Reproduce
Profilers/A-B harnesses (agent scripts, run in `flowwam-robotwin` via `flowwam_p8_run.sh`):
`flowwam_p8_profile.py` (per-stage), `flowwam_p8_dit_breakdown.py` (intra-forward + SDPA probe +
RoPE A/B), `flowwam_p8_cache_ab.py`, `flowwam_p8_quant_ab.py`, `flowwam_p8_compile_ab.py`,
`flowwam_p8_asym_ab.py`; closed-loop: `flowwam_p8_val_server.sh` (FLOW_DS) + `flowwam_cl_eval.sh`.
Shipped-stack validation: `flowwam_p8_stack_ab.py` (imports the shipped `flowwam_opt.py`; toggle
`FLOWWAM_CACHE`/`FLOWWAM_COMPILE`) and closed-loop `flowwam_opt_clval.sh` (baked route end-to-end).
