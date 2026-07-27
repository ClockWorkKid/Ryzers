# W4A6 quantization of the distilled MolmoAct2 vision tower — full dev-flow report

This report covers the **complete development flow** for two deployable W4A6
(4-bit weight, 6-bit activation) quantized vision-tower students for MolmoAct2:

```
teacher SigLIP2 ViT  ──distill+finetune──▶  100×-smaller fp32 student
                                                   │
                                       ┌───────────┴───────────┐
                                       ▼                       ▼
                              PTQ (Brevitas)            PTQ (Brevitas)
                              W8A8 / W4A6               W8A8 / W4A6
                                       │                       │
                                       ▼                       ▼
                              QAT recovery (W4A6)      QAT recovery (W4A6)
```

Two student families are shipped:
- **conv (`cnn`)** — a convolutional, FPGA-friendly encoder (KV260 dataflow target).
- **nano (`tinyvit`)** — a narrow attention encoder ("nano SigLIP").

All numbers below are on real LIBERO data with the real distilled weights. The
FPGA/KV260 dataflow back-end and its reference recipe are kept out of this
package (local-only, not upstreamed).

---

## 1. fp32 student (distill + finetune)

The students are ~**100× fewer GFLOPs** than the teacher SigLIP2 tower and are
plugged into the frozen MolmoAct2 policy (2×2 pool → vision→LLM projector →
Qwen3 backbone → flow-matching action head). Closed-loop LIBERO success (4
suites × 20 episodes) is the deployment metric of record.

| variant | closed-loop mean (fp32 student) |
|---|---|
| conv (`cnn`) | 86.2 |
| nano (`tinyvit`) | 92.5 |

Checkpoints: `cnn_fpga_full.pt`, `siglip_nano_full.pt` (see `WEIGHTS.md`).

---

## 2. Post-training quantization (PTQ) — Pareto sweep

Brevitas fake-quant is applied to the student (`QuantLinear` + `QuantConv2d`,
`QuantAttnBlock`/QSDPA), then PTQ-calibrated on real LIBERO frames. Seam
fidelity is the cosine of the student seam features vs the fp32 student, on
clean/un-augmented held-out frames.

| variant | W8A8 cos | W6A6 cos | W4A8 cos | **W4A6 cos (target)** | rel-L2 @ W4A6 |
|---|---|---|---|---|---|
| conv (`cnn`)  | 0.9967 | 0.9433 | 0.8793 | **0.8261** | 0.610 |
| nano (`tinyvit`) | 0.9773 | 0.8408 | 0.9367 | **0.7920** | 0.627 |

- **W8A8 PTQ is near-lossless** — a safe high-fidelity fallback.
- **W4A6 PTQ regresses** → QAT recovery required (Section 3).
- Sensitivity differs: the conv model is limited by 4-bit **weights**
  (W6A6 > W4A8); the attention model by 6-bit **activations** (W4A8 > W6A6).

Closed-loop confirms the PTQ collapse at W4A6:

| variant | W8A8 PTQ | W4A6 PTQ |
|---|---|---|
| conv (`cnn`) | 90.0 | 56.2 |
| nano (`tinyvit`) | 97.5 | 51.2 |

---

## 3. QAT recovery (W4A6)

Quantization-aware distillation of the fake-quant student against the **frozen
fp32 student teacher** (seam cosine + normalized-MSE loss), on LIBERO, with a
**short, cosine-decayed** schedule from the PTQ-calibrated init.

Clean held-out seam fidelity (apples-to-apples with the PTQ sweep):

| variant | W4A6 PTQ | **W4A6 + QAT** | min cos | recovery |
|---|---|---|---|---|
| conv (`cnn`)  | 0.8250 | **0.9419** | 0.9307 | +0.117 |
| nano (`tinyvit`) | 0.7936 | **0.9608** | 0.9404 | +0.167 |

**Closed-loop (the metric that matters):**

| variant | fp32 | W8A8 PTQ | W4A6 PTQ | **W4A6 QAT** | Δ vs fp32 |
|---|---|---|---|---|---|
| conv (`cnn`)  | 86.2 | 90.0 | 56.2 | **95.0** | **+8.8** |
| nano (`tinyvit`) | 92.5 | 97.5 | 51.2 | **93.8** | **+1.2** |

The short cosine-LR LIBERO QAT recovers the aggressive 4-bit-weight target to
**at or above the fp32 student** in closed loop, not just the W8A8 fallback.

Full per-suite breakdown:

### conv (`cnn`)
| stage | spatial | object | goal | long | mean | Δ vs fp32 |
|---|---|---|---|---|---|---|
| fp32 (baseline) | 95 | 85 | 80 | 85 | 86.2 |  |
| W8A8 PTQ | 95 | 85 | 85 | 95 | 90.0 | +3.8 |
| W4A6 PTQ | 75 | 35 | 75 | 40 | 56.2 | -30.0 |
| **W4A6 QAT** | 90 | 95 | 95 | 100 | **95.0** | **+8.8** |
| W4A6 re-distill (DROID mix, 3ep) | 50 | 75 | 70 | 45 | 60.0 | -26.2 |
| W4A6 re-distill + QAT-ft (LIBERO, 3ep) | 65 | 25 | 80 | 20 | 47.5 | -38.8 |

### nano (`tinyvit`)
| stage | spatial | object | goal | long | mean | Δ vs fp32 |
|---|---|---|---|---|---|---|
| fp32 (baseline) | 90 | 95 | 90 | 95 | 92.5 |  |
| W8A8 PTQ | 95 | 100 | 100 | 95 | 97.5 | +5.0 |
| W4A6 PTQ | 90 | 30 | 70 | 15 | 51.2 | -41.2 |
| **W4A6 QAT** | 100 | 100 | 95 | 80 | **93.8** | **+1.2** |
| W4A6 re-distill (DROID mix, 3ep) | 75 | 80 | 95 | 75 | 81.2 | -11.2 |
| W4A6 re-distill + QAT-ft (LIBERO, 3ep) | 70 | 35 | 75 | 45 | 56.2 | -36.2 |

---

## 4. Key finding — the seam proxy is anti-correlated with closed-loop success

We also ran a much heavier recipe: full-epoch (3 epochs) re-distillation on a
**DROID+LIBERO mix** with a constant LR, followed by a full-epoch LIBERO QAT
fine-tune. It **optimized the static seam proxy harder yet regressed closed-loop
badly.** Clean held-out seam fidelity vs closed-loop, conv (`cnn`):

| model | clean seam cos | clean rel-L2 | closed-loop |
|---|---|---|---|
| W4A6 QAT (short, LIBERO, cosine-LR) | 0.941 | 0.863 | **95.0** |
| W4A6 re-distill (3ep, DROID mix) | 0.953 | 0.314 | 60.0 |
| W4A6 re-distill + QAT-ft (3ep) | 0.959 | 0.369 | 47.5 |

**The best closed-loop model has the worst static seam fidelity.** Root cause:
(1) the seam-distillation proxy is misaligned with control; (2) long / constant-LR
training + DROID mixing + heavy augmentation drift the encoder **away** from the
fp32-student init that the frozen policy was co-adapted to, which hurts behaviour
on the rollout distribution (which differs from static demo frames); (3) more
proxy optimization = worse policy.

**Recommendation (shipped):** the **short, cosine-LR, LIBERO-only W4A6 QAT** is
the Pareto winner and is the deployable checkpoint. Select quantized models on
**closed-loop success**, not seam cosine.

---

## 5. QONNX export

Both winners export to verified QONNX (Brevitas `export_qonnx`, dynamo→trace
auto, qonnx cleanup, parity vs PyTorch fake-quant, target cosine ≥ 0.99).

| variant | precision | total nodes | Quant | LayerNorm | Erf(GELU) | Conv | MatMul | Softmax |
|---|---|---|---|---|---|---|---|---|
| conv (`cnn`) | W4A6 QAT | 316 | 68 | 17 | 16 | 16 | 18 | 0 |
| conv (`cnn`) | W8A8 PTQ | 316 | 68 | 17 | 16 | 16 | 18 | 0 |
| nano (`tinyvit`) | W4A6 QAT | 205 | 60 | 9 | 4 | 0 | 26 | 4 |
| nano (`tinyvit`) | W8A8 PTQ | 205 | 60 | 9 | 4 | 0 | 26 | 4 |

The `LayerNormalization` + `Erf` (GELU) ops are what a downstream dataflow
streamlining pass must absorb.

---

## 6. What ships here

- `quant/` — the full pipeline (fake-quant, PTQ + Pareto, QAT, closed-loop eval,
  QONNX export, seam eval). See `quant/README.md`.
- `WEIGHTS.md` + `download_weights.sh` — the full dev-flow checkpoints
  (distill+ft → W8A8/W4A6 PTQ → W4A6 QAT, plus the re-distill negative-result
  history) for **both** variants, hosted externally (weights are not committed).
- `REPRODUCE.md` — end-to-end commands.

The FPGA/KV260 dataflow back-end and its reference recipe are intentionally **not**
included (kept local, not upstreamed).
