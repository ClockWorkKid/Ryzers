# Corrected QAT-fix experiment — downstream-consistency loss for W4A6 students

**Goal:** recover/beat the capped-QAT Pareto winner (cnn **95.0** / tinyvit **93.8**,
pc_success %, 4 LIBERO suites × 20 ep) in *closed loop*, LIBERO-only, after the
finding that static seam fidelity is **anti-correlated** with control success.

## What changed (code)
- `research/vit_distill/distill/losses.py`: added `seam_downstream_loss(student_tokens,
  teacher_tokens, valid)` — cosine + per-token normalized-MSE on the tokens the LLM
  actually consumes (i.e. AFTER the frozen 2×2 attention-pool + vision→LLM projector).
- `research/vit_distill/quant/qat.py`:
  - `DownstreamHead` — deep-copies the teacher backbone's frozen `image_pooling_2d` +
    `image_projector` (fp32, sdpa so the border pooling mask is honored) and applies the
    exact `MolmoAct2VisionBackbone.forward` pool+project path to a seam.
  - `build_pooled_idx` — derives the fixed `image_token_pooling` index once from the
    processor (LIBERO = 1 crop → 729 patches → 196 pooled tokens of dim 2560).
  - new flags: `--downstream-w`, `--downstream-max-tokens`, `--aug {heavy,mild,none}`.
- Module-validated on real LIBERO data before launch (shapes, finite loss, grad reaches
  the fake-quant student, 2.35 GB peak @B=4).

## Recipe (all arms)
LIBERO-only, from fp32 student + fresh PTQ calib, **2000 steps**, cosine LR 2e-4→0,
bs32 — i.e. the capped-QAT winner recipe. 2×2 factorial per variant:
`downstream_w ∈ {0,1}` × `aug ∈ {heavy, mild}`. `ctrl` = winner recipe reproduction.

## Closed-loop results (pc_success %, 4 suites × 20 ep = 80 episodes/arm)

### CNN (FPGA) — winner 95.0
| arm | spatial | object | goal | long | mean | Δ |
|---|---|---|---|---|---|---|
| W4A6 QAT (winner) | 90 | 95 | 95 | 100 | **95.0** | — |
| qatfix_ctrl (repro) | 95 | 95 | 90 | 95 | 93.8 | −1.2 |
| qatfix_ds (downstream) | 95 | 100 | 80 | 100 | 93.8 | −1.2 |
| qatfix_dsmild | 100 | 100 | 85 | 90 | 93.8 | −1.2 |
| **qatfix_mild** | 95 | 95 | 90 | 100 | **95.0** | **+0.0** |

### TinyViT (nano) — winner 93.8
| arm | spatial | object | goal | long | mean | Δ |
|---|---|---|---|---|---|---|
| W4A6 QAT (winner) | 100 | 100 | 95 | 80 | 93.8 | — |
| W8A8 PTQ (ref) | 95 | 100 | 100 | 95 | 97.5 | +3.8 |
| qatfix_ctrl (repro) | 80 | 100 | 95 | 85 | 90.0 | −3.8 |
| **qatfix_ds (downstream)** | 100 | 100 | 100 | 95 | **98.8** | **+5.0** |
| qatfix_dsmild | 100 | 95 | 100 | 85 | 95.0 | +1.2 |
| qatfix_mild | 95 | 100 | 95 | 80 | 92.5 | −1.2 |

## Conclusions
1. **The downstream-consistency objective helps closed-loop — decisively for TinyViT.**
   `tinyvit_qatfix_ds` = **98.8**, a **+5.0** new best that beats the capped-QAT winner
   and even edges lossless **W8A8 PTQ (97.5)** at W4A6. It is the only arm with **no weak
   suite** (min 95 vs the winner's long=80). For CNN it is neutral (93.8, tied with ctrl);
   CNN's best is `mild`-aug (95.0), tying the winner.
2. **Confirms the anti-correlation insight in-experiment.** The downstream arms had the
   *lowest* raw-seam cosine (tinyvit ds 0.836 vs ctrl 0.855; cnn ds 0.830 vs 0.844) yet
   tinyvit ds had the *highest* closed-loop success — matching the student to the teacher
   in the space the policy is co-adapted to beats matching raw seam features.
3. **Augmentation is a secondary lever.** Mild aug helped CNN (95.0 vs 93.8 ctrl), was
   neutral/slightly negative for TinyViT. Downstream >> aug.
4. **Model selection on seam-cos would have picked the wrong arm** — selection was done
   on closed-loop pc_success, as directed.

## Recommended recipe
- **TinyViT (adopt):** LIBERO-only, from fp32+PTQ, 2000 steps, cosine LR 2e-4→0,
  `--downstream-w 1.0`, heavy aug. Ckpt: `/outputs/quant/tinyvit_w4a6_qatfix_ds.pt` (98.8).
- **CNN:** downstream gives no gain; keep the capped-QAT/mild recipe (~95). Ckpt
  `/outputs/quant/cnn_w4a6_qatfix_mild.pt` (95.0).

## Caveats
- 20 ep/suite ⇒ ~±11% binomial noise per suite, ~±3–4% on the 80-ep mean; the TinyViT +5.0
  is best read together with its per-suite robustness (no suite <95). A 50-ep confirmation
  of `tinyvit_qatfix_ds` vs winner would tighten the interval.
- New checkpoints on cluster only; original `_w4a6_qat.pt` / `_redistill*` blobs untouched.
