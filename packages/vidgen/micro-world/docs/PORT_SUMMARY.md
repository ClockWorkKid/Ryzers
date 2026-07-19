# Micro-World Ryzer — Port Summary (progress log)

Upstream: github.com/AMD-AGI/Micro-World @ `e63369c` · bases: `Wan-AI/Wan2.1-T2V-1.3B` (t2v/t2w) /
`Wan-AI/Wan2.1-I2V-14B-480P` (i2v/i2w) + `amd/Micro-World-{T2W,I2W}` · target: AMD Strix Halo (Ryzen
AI Max+ 395, `gfx1151`), ROCm 7.2.2 · branch `wm-micro-world` off `benchmark`. Scope: **inference**
(training / eval — DeepSpeed ZeRO, VPT IDM + GameWorldScore — out).

## Phase 1 — baseline image + full import smoke ✅ (Gate 1)
- Direct PyTorch → ROCm port: base image ROCm torch (2.10.0+rocm7.2.2) preserved; upstream's curated
  **inference** deps installed under a `torch/torchvision/numpy` `PIP_CONSTRAINT` (`diffusers==0.34.0`,
  `transformers>=4.46.2,<5`) so nothing pulls a CUDA wheel. Training-only heavy deps omitted.
- **ROCm patch (rule 2.1), minimal + idempotent** (`patches/rocm_port.py`): the CLIP image encoder's
  `AttentionPool` hard-calls `flash_attention()` (asserts a flash-attn wheel gfx1151 lacks) → added a
  torch-SDPA fallback there. The DiT's own `attention()` dispatcher already SDPA-falls-back, so no
  other change. flash-attn intentionally not installed.
- **Compat fixes:** `transformers` capped `<5` (diffusers 0.34.0 imports `FLAX_WEIGHTS_NAME`, removed
  in transformers 5.x); `apex` uninstalled (its `FusedRMSNorm` mishandles half precision — mirrors the
  Latte port). Build asserts `torch.version.hip` so a clobbered non-ROCm torch fails early.
- Result: `micro-world` builds; `test.py` PASS on `Radeon 8060S` — all modules import
  (`AutoencoderKLWan`, `WanT5EncoderModel`, `WanTransformer3DModel`, `WanActionControlNetModel`,
  `WanActionAdaLNModel`, `CLIPModel`, the 4 pipelines, utils), on-device SDPA attention + action-list
  parsing validated (rule 2).

## Phase 2 — weights + per-component load ✅ (Gate 2)
- Weights fetched at runtime (rule 8) via `scripts/download_checkpoints.sh` (WHICH selects subset).
  The large Wan2.1 bases were **reused from the VERA workspace via hardlink** (`sudo cp -al`) to avoid
  re-download; `amd/Micro-World-{T2W,I2W}` (transformer shards + lora) downloaded. Layout mirrors the
  upstream README so the unmodified examples resolve their paths.
- **Per-component load tests** (`scripts/component_load_test.py`, partial-weight, rule 2): t2v
  transformer **1.42B** and t2w controlnet **2.16B** fit full-resident within the box's **~47 GB
  GPU-addressable memory** (32 GB VRAM carveout + GTT; the 96 GB is total system RAM); the 14B
  i2v (**16.4B**) is tight and i2w adaln (**18.3B**) needs **`model_cpu_offload`** — set as the
  per-mode default for the image paths.

## Phase 3 — all 4 examples runnable ✅ (Gate 3)
- Unified env-driven runner `demos/mw_generate.py` (+ thin `demos/demo_*.sh` wrappers) reproduces
  the four `examples/wan2.1/predict_*` scripts without editing upstream in place. Action schedule is
  auto-clamped to the effective (VAE-grid-rounded) frame count so short validation runs work with the
  default schedule (no fragile env override).
- All four ran headless on `gfx1151`, coherent + action-following (artifacts mirrored small to
  `artifacts/micro-world/`, rule 4):
  - **t2v** (1.3B, full-resident): 49 frames / 20 steps → single-column MP4 (rule 2.b).
  - **t2w** (controlnet 1.3B): 81 frames / 20 steps, action HUD-overlaid, camera follows keys.
  - **i2v** (14B, cpu-offload): 25 frames / 15 steps → two-column reference | generated (rule 2.b).
  - **i2w** (14B adaln + lora, cpu-offload): 49 frames / 15 steps, ~105 s/step (≈25 min denoise) →
    two-column reference | generated.

## Phase 4 — aggressive optimization ✅ (benchmark recipe)
- **Component profile + A/B harness** `scripts/opt_ab_microworld.py` (CUDA-event forward timers;
  loads the cheap t2v-1.3B once, sweeps levers in-process — rule 0.5). Full write-up +
  numbers: `RUNTIME_OPTIMIZATION.md`; raw `artifacts/micro-world/mw_opt_ab.json`; shared-playbook
  ledger entry in `docs/RUNTIME_OPTIMIZATION_PLAYBOOK.md`.
- **Key finding (contrast with VERA):** Micro-World is **DiT-denoise-loop bound — 92.7 % (52.2 s of
  56.3 s)**; WAN VAE conv3d 5.5 %, T5 1.8 %. So the movers are step-skip / step-count, not the conv
  override (which was VERA's 8.8× win on its conv-dominated stack).
- **Shipped (baked in `demos/mw_generate.py`), all Gate-3 validated:**
  - **conv3d override** (`cudnn.enabled=False`, `apply_gfx1151_speedups()`) — VAE decode **1.49×**
    (grows on the 49-frame image paths); free, default-on, kill-switch `MW_DISABLE_CUDNN=0`.
  - **built-in TeaCache** default-on, per-mode thresholds (t2v/i2v 0.10, t2w/i2w 0.20): thr 0.10 →
    **1.51× @ PSNR 23.0 dB**, thr 0.20 → 2.09× @ 20.3 dB.
  - **UniPC 30 steps** default (`NUM_STEPS=20` → 1.45× at solid quality; 10 → 2.62× but soft).
  - bf16 autocast already present upstream (captured). `cfg_skip` wired (`CFG_SKIP_RATIO`) as opt-in.
- **Evaluated, not adopted:** cross-attn K/V cache — **bit-exact (max|Δ|=0.0) but 1.00×** (DiT is
  video-self-attn bound; cross-attn context is small). Prototype retained in the harness (`kv=True`).

## Phase 5 — ship
- Package redeployed with the optimized runner; docs (`README.md`, this summary,
  `RUNTIME_OPTIMIZATION.md`) written; scratch pruned.
- **Next:** dev full manual repro test → local commit on `wm-micro-world` → STOP before push (rule
  0.3), await approval, then push + optional upstream PR.
