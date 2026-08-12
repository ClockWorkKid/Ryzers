# GR-1 inference optimization on AMD Strix Halo (gfx1151, ROCm)

Phase 4 of the GR-1 port. Goal: speed up GR-1's per-step inference on the Radeon 8060S iGPU
(gfx1151) **without regressing action quality**. Every config is scored on both axes with one
harness (`demos/demo_bench.py`), and the winner is confirmed to hold the closed-loop CALVIN task
success from phase 3 (8/8 on the debug validation set).

## What dominates the cost
`GR1CalvinEvaluation.step` reruns the **entire `seq_len=10` sequence** every timestep (no KV cache):
two batched **MAE ViT-B/16** passes (static + gripper cameras, batch 10 × 197 tokens × 12 layers),
then the Perceiver resampler and the GPT-2 trajectory backbone. The ViT is the bottleneck, so the
optimizations target (a) precision for all matmuls and (b) attention specifically in the ViT.

## Optimization layer
`demos/gr1_optim.py` applies three **opt-in, env-toggled** optimizations on top of the *unmodified*
upstream graph (rule 2.1 - no invasive edits; stays a clean opt-in wrapper for an upstream PR):

| env var | values | effect |
|---|---|---|
| `GR1_AMP` | `off` / `bf16` / `fp16` | `torch.autocast` dtype for the policy forward |
| `GR1_SDPA` | `0` / `1` | `torch.nn.functional.scaled_dot_product_attention` in the MAE ViT |
| `GR1_FLASH` | `0` / `1` | pin the ROCm-native FLASH (AOTriton) SDPA backend (default `1`) |
| `GR1_COMPILE` | `0` / `1` | `torch.compile` the policy (`GR1_COMPILE_MODE`, default `default`) |

SDPA in the ViT is numerically equivalent to the vendored path (default SDPA scale = `head_dim**-0.5`
= the module's scale; the discarded attention weights are unused on the forward path). autocast keeps
softmax/layernorm/reductions in fp32.

### ROCm-native flash attention on gfx1151
There is **no standalone flash-attention** for this GPU: the `flash_attn` / composable-kernel (CK)
library targets CDNA (MI200/MI300) and has no RDNA3.5 (gfx1151) kernels (`import flash_attn` ->
`ModuleNotFoundError`). ROCm's native flash attention is exposed **through torch SDPA's AOTriton
FLASH backend**. On-device backend probe (MAE ViT shape `(20,12,197,64)`):

| dtype | FLASH | mem-efficient | math | torch auto-select |
|---|---|---|---|---|
| fp16 | **0.181 ms** | 0.183 ms | 3.09 ms | 0.184 ms (picks FLASH) |
| bf16 | **0.189 ms** | 0.190 ms | 3.08 ms | 0.189 ms (picks FLASH) |
| fp32 | *no kernel* | 1.62 ms | 2.44 ms | 1.62 ms (mem-efficient) |

So auto-select already lands on the flash kernel for fp16/bf16, but `GR1_FLASH=1` **pins** it
explicitly via `torch.nn.attention.sdpa_kernel([FLASH, EFFICIENT, MATH])`: we deterministically run
the ROCm flash kernel and never silently regress to the ~17× slower math path, while the mem-efficient
entry safely catches fp32 (where no flash kernel exists) instead of erroring. Pinning is marginally
faster than auto (fp16+sdpa: 56.5 ms flash-pinned vs 57.1 ms auto) and preserves closed-loop 8/8.

## Results (Radeon 8060S / gfx1151, torch 2.10.0+rocm7.2.2, per-step, 32 timed steps + 8 warmup)

Quality: open-loop arm MAE vs CALVIN ground-truth `rel_actions` (baseline fp32 = **0.0421**),
gripper match, and closed-loop task success on the 8 CALVIN debug validation tasks (phase 3).

| config | mean ms | fps | speedup | arm MAE | closed-loop |
|---|---|---|---|---|---|
| **fp32 baseline** | 389.7 | 2.57 | 1.0× | 0.0421 | 8/8 |
| sdpa (fp32) | 562.8 | 1.78 | 0.69× | 0.0421 | - |
| bf16 | 74.5 | 13.42 | 5.2× | 0.0421 | - |
| fp16 | 69.3 | 14.43 | 5.6× | 0.0421 | - |
| bf16 + sdpa | 59.2 | 16.89 | 6.6× | 0.0420 | 7/8 |
| **fp16 + sdpa** (default) | **57.5** | **17.41** | **6.8×** | 0.0421 | **8/8** |
| bf16 + sdpa + compile | 47.3 | 21.14 | 8.2× | 0.0426 | - |
| **fp16 + sdpa + compile** | **45.0** | **22.23** | **8.7×** | 0.0426 | **8/8** |

## Findings
1. **Precision (AMP) is the dominant win** - ~5× from fp16/bf16 alone, with *zero* open-loop quality
   loss (arm MAE unchanged). Most GR-1 step time is matmuls that autocast moves to low precision.
2. **SDPA only helps in low precision.** In fp32 it is *slower* (0.69×) on gfx1151 - the AOTriton
   flash path isn't tuned for fp32 and falls back. In fp16/bf16 it adds ~1.2× over AMP alone.
3. **fp16 beats bf16 here.** fp16 has 10 mantissa bits vs bf16's 7; inference activations are small
   so fp16's narrower exponent range is not a problem. The extra precision preserves closed-loop
   robustness on the hardest long-horizon task: `bf16+sdpa` drops to **7/8** (one
   `lift_blue_block_slider` variant diverges after ~120 steps), while `fp16+sdpa` holds **8/8** at
   equal or better speed.
4. **torch.compile adds ~1.25×** on top (shapes are static after the buffer fills, so it traces once
   with no per-step recompiles), for a one-time ~30-60 s first-step compile cost.

## Recommendation
- **Default fast config: `GR1_AMP=fp16 GR1_SDPA=1 GR1_FLASH=1`** - 6.8× faster (2.6 → 17.4 fps, ROCm
  flash-pinned), arm MAE identical to fp32, closed-loop 8/8. Robust (no compile fragility).
- **Max throughput (opt-in): add `GR1_COMPILE=1`** - 8.7× (→ 22.2 fps), also 8/8, if the one-time
  first-step compile stall is acceptable.
- Keep fp32 (all toggles off) as the numerical reference.

The image bakes the recommended defaults (`config.yaml`: `GR1_AMP=fp16 GR1_SDPA=1 GR1_FLASH=1`), so a
plain `ryzers run --name gr1 /ryzers/demos/demo_calvin.sh` (closed-loop) or `/ryzers/demos/demo_bench.sh`
(benchmark) already runs the 6.8× fast path. Add `GR1_COMPILE=1` for the 8.7× config, or set
`GR1_AMP=off GR1_SDPA=0` to fall back to the fp32 numerical reference (the `gr1_optim.py` module itself
defaults every toggle off, so the fp32 baseline is what you get if the env is unset).
