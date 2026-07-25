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
| **Dual-stream asymmetry** (flow DS=2, unique) | model-specific | **1.61×** | closed-loop 3/5 (60%) vs 3/3 baseline | **too lossy at DS=2** |
| fp64→fp32 RoPE | conditional | ~1.02× (est.) | max\|Δ\|=0.18, not exact | skip (low ROI) |

### Why W8A8 is a regression here (correction to the playbook hypothesis)
FlowWAM's hot loop *is* big-GEMM (ffn=14336, run 26×/replan), which suggested §2.5 quant should
pay off — but it does **not** on this ROCm stack. Isolated `torch._int_mm` at M=1560:
`ffn_up 3072→14336 = 0.28×`, `ffn_down 14336→3072 = 1.17×`, `qkvo 3072→3072 = 0.57×`. The per-token
dequant over the 14336-wide output is bandwidth-bound and `_int_mm` isn't tuned for the up-proj
shape. **Big-GEMM ≠ quant win on gfx1151.**

### FlowWAM-unique lever: dual-stream asymmetry
The flow stream shares the DiT blocks and is a secondary motion channel. Running it at half spatial
resolution (480→120 tokens/frame) shrinks the joint self-attention and quarters the flow-side
qkvo/FFN/RoPE — **no model_fn change needed** (the flow grid/RoPE/head derive from the flow-latent
shape). Result: **1.61× on the DiT loop**, but the RGB latent moved (cos 0.92) and closed-loop
`beat_block_hammer` dropped **100% → 60% (3/5)**. Server-side, env-gated via `FLOW_DS`
(`agent_scripts/flowwam_p8_val_server.sh`). **Open:** a milder reduction (single-dim downsample, or
~0.75× area) or a short finetune at reduced flow res to recover success — this is the highest-upside
remaining direction and is genuinely FlowWAM-specific.

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
- **Open**: dual-stream asymmetry at a validated sweet-spot DS (biggest upside, ~1.6× but lossy at
  DS=2 — deliberately NOT default; experimental `FLOW_DS` knob only); fewer NFE / distillation
  (§2.6, out of P8 scope — the known knob).

## Reproduce
Profilers/A-B harnesses (agent scripts, run in `flowwam-robotwin` via `flowwam_p8_run.sh`):
`flowwam_p8_profile.py` (per-stage), `flowwam_p8_dit_breakdown.py` (intra-forward + SDPA probe +
RoPE A/B), `flowwam_p8_cache_ab.py`, `flowwam_p8_quant_ab.py`, `flowwam_p8_compile_ab.py`,
`flowwam_p8_asym_ab.py`; closed-loop: `flowwam_p8_val_server.sh` (FLOW_DS) + `flowwam_cl_eval.sh`.
Shipped-stack validation: `flowwam_p8_stack_ab.py` (imports the shipped `flowwam_opt.py`; toggle
`FLOWWAM_CACHE`/`FLOWWAM_COMPILE`) and closed-loop `flowwam_opt_clval.sh` (baked route end-to-end).
