# Distilling, quantizing and recovering a robot-policy vision encoder down to 2-bit weights: MolmoAct2

---

## Abstract

We study how far the precision of a robot-policy vision encoder can be reduced while preserving closed-loop task success. The objective is **2-bit weights**, because on the target FPGA (Field-Programmable Gate Array) 2-bit multiplies map onto abundant look-up tables (LUTs) rather than the scarce dedicated multipliers (DSP blocks). We test the hypothesis that accuracy lost to quantization can be recovered by increasing model capacity ("grow-to-compensate"), sweeping three student sizes (S/M/L) against eight weight/activation precisions. The recipe has three stages: (1) full-precision knowledge distillation of each student from a frozen teacher; (2) learned-step-size quantization with a short quantization-aware screen; (3) a downstream-consistency fine-tune for accuracy recovery. The medium and large students retain **≥97% closed-loop success with 2-bit weights** (L@W2A6 = 98.3%, L@W2A4 = 97.5%, M@W2A6 = 97.5%), while the small student achieves a reasonable performance of up to 95.8%.

---



## 1. Notation and metrics

- **fp32 / bf16.** Numeric formats. **fp32** = 32-bit floating point (full precision, the reference). **bf16** = bfloat16, a 16-bit float used for the frozen teacher to save memory.
- **W*w*A*a* notation.** A precision configuration with *w*-bit **weights** (the model's learned parameters) and *a*-bit **activations** (the tensors flowing through the model at run time). E.g. **W2A6** = 2-bit weights, 6-bit activations. Lower bit-widths reduce hardware cost but shrink numeric range.
- **CL (closed-loop) success.** The primary metric: the model is placed in the full perceive→act→observe feedback loop and we report the fraction of tasks completed. This captures error accumulation that a simple cosine similarty would miss.
- **LIBERO / suites.** The manipulation benchmark. Four **suites** (task families): `object`, `spatial`, `goal`, and `10` (the long-horizon suite). We evaluate **30 episodes/suite × 4 suites = 120 episodes** per configuration and report the mean.
- **Target.** A configuration "passes" if CL ≥ **97%** (outlined cells in the heatmap).
- **Params / GMACs.** Cost measures. **Params** = number of learned values (memory). **MAC** = one multiply-accumulate; **GMACs** = billions of MACs per forward pass (compute). FLOPs (floating-point operations) = 2 × MACs.
- **Wilson 95% CI.** A confidence interval (CI) is the plausible range of the true success rate given finite episodes; the Wilson interval is the appropriate estimator for a binomial proportion. The **± value** under each heatmap cell is its half-width (≈±2–4 points at n=120), so cells within a few points are statistically tied.

---



## 2. Models



### 2.1 Teacher

A frozen full-precision (fp32, executed in bf16) vision tower from the MolmoAct2 robot policy (SigLip2). The features we match are taken at the "**seam"**  : the interface where vision features would pass into the rest of the policy, there is a pooling layer at the seam.

### 2.2 Students

All students are **TinyViT** encoders  : a compact **ViT** (Vision Transformer, the patch-based attention architecture). They share **8 attention heads** and an **MLP ratio of 2.0** (each block's feed-forward width is 2× the embedding width) and differ only in embedding width and depth:


| size  | embedding dim | transformer blocks | params (fp32) | GMACs / image tile |
| ----- | ------------- | ------------------ | ------------- | ------------------ |
| **S** | 240           | 4                  | **2.73 M**    | **2.87**           |
| **M** | 320           | 6                  | **6.10 M**    | **6.30**           |
| **L** | 416           | 8                  | **12.62 M**   | **12.49**          |


L is ~4.6× the parameters and ~4.3× the compute of S. Cost is per single viewpoint of 729 tokens (14×14-pixel patches); params are exact from the checkpoints, linear-layer compute is measured by PyTorch's dispatch-level counter, and attention compute uses the standard `2·L·T²·D` closed form (see `cost_smL_prequant.png`). Sizes are registered in `research/vit_distill/quant/variants.py` as `tinyvit_s/m/l`.

---



## 3. Stage 1  : Full-precision distillation

**Knowledge distillation (KD)** trains each student to reproduce the teacher's **seam** features from the same inputs.

**Loss.** Equal-weighted sum of two seam terms:

- **seam cosine loss**  : aligns feature *direction* (`seam_cosine_loss`);
- **normalized MSE** (Mean Squared Error)  : aligns feature *magnitude*, scale-normalized (`seam_norm_mse_loss`).

**Hyperparameters** (`distill_fp32.py`): optimizer **AdamW**, learning rate **1e-3**, weight decay **0.05**, **cosine-annealing** schedule, batch size **32**, teacher in bf16, DistributedDataParallel across 4 GPUs.

**Runs.**

- **S** reuses an existing distilled checkpoint (`siglip_nano_full.pt`), which is also the basis of the prior high-precision reference (W4A6 = 98.8% CL).
- **M** and **L** are distilled for ~40,480 steps, reaching seam cosine similarity to the teacher of **0.902 (M)** and **0.921 (L) before finetuning**.

---



## 4. Stage 2  : Quantization initialization (PTQ) and learned scales (LSQ)

We use **Brevitas** (a PyTorch quantization library) to insert **fake-quant** operators: tensors are rounded to the target grid on the forward pass but kept in fp32 internally so gradients still flow, via the **STE** (Straight-Through Estimator, which treats the non-differentiable rounding as the identity in the backward pass).

**PTQ initialization.** Before any training we apply **PTQ** (Post-Training Quantization): calibrate the quantization **scale** (real-value range per step) from statistics over **16 batches**. Statistics-based (MAX) scaling is adequate at 4–8 bits but degenerate at 2-bit weights (only 4 symbols, dominated by outliers), so it is used only as an initialization.

**LSQ for aggressive weights.** For **≤3-bit weights** the per-channel step size is made a **learned parameter**  : **LSQ** (Learned Step-size Quantization)  : and trained jointly with the weights. Scales are **per-channel** for weights (one per output channel) and **per-tensor** for activations. LSQ is the key enabler of trainable 2-bit weights; quantizers are defined in `research/vit_distill/quant/quant_student.py` for bit-widths {2, 3, 4, 6, 8}.

**Screen.** A short **QAT** (Quantization-Aware Training) pass of **2,000 steps** with the auxiliary recovery loss disabled (downstream weight = 0; see §5) triages all 24 configurations before committing to the full fine-tune.

---



## 5. Stage 3  : Downstream-consistent QAT fine-tune (accuracy recovery)

Matching the seam alone leaves residual error that propagates into the policy. We add a **downstream-consistency loss**: both the quantized student and the fp32 student are passed through a **frozen** miniature of the next stage (a 2×2 attention pool plus the vision→language projector), and we penalize their disagreement (`seam_downstream_loss`). Its strength is the **DSW** (Downstream-consistency Weight):

- **DSW = 0**  : off (Stage-2 screen).
- **DSW = 1.0**  : on (recovery fine-tune, checkpoints tagged `qatfix_ds`).

**Final fine-tune (**`qatfix_ep`**).** Each of the 24 configurations is fine-tuned for **2 epochs** from its `qatfix_ds` checkpoint with the following settings (`qat.py`, `sweep_qat_epoch.sbatch`):


| setting                 | value                                                                           |
| ----------------------- | ------------------------------------------------------------------------------- |
| loss                    | cosine (w=1.0) + normalized-MSE (w=1.0) + downstream (**DSW=1.0**, ≤256 tokens) |
| optimizer               | AdamW, weight decay 0.05                                                        |
| learning rate           | **5e-5, constant** (chunk-friendly)                                             |
| batch size              | **32** (S, M) / **16** (L)                                                      |
| target steps (2 epochs) | **32,388** (S, M) / **64,776** (L)                                              |
| augmentation            | heavy                                                                           |
| calibration             | skipped (resume from already-calibrated `qatfix_ds`)                            |
| checkpointing           | every 1,000 steps                                                               |


**Compute.** Single node, 4 GPUs (AMD Instinct MI210-class, 64 GB each), multiple arms packed per node; long runs are split into ≤4-hour chunks that checkpoint and auto-resubmit until each arm reaches its target step count.

---



## 6. Evaluation protocol

Each of the 24 `qatfix_ep` checkpoints is patched into the MolmoAct2 vision backbone and run **closed-loop** on all four LIBERO suites, **30 episodes/suite (120 total)**. The evaluator reads the weight/activation bit-widths and variant from the checkpoint and reconstructs the quantized encoder accordingly. Per-cell Wilson 95% CIs are computed at n=120.

---



## 7. Results

Closed-loop success (%) per size × precision (n=120; bold = ≥97% target):


| bits | S (2.73M) | M (6.10M) | L (12.62M) |
| ---- | --------- | --------- | ---------- |
| W4A6 | 95.0      | **97.5**  | 96.7       |
| W4A4 | 95.8      | **97.5**  | **99.2**   |
| W3A6 | 95.0      | 95.8      | **99.2**   |
| W3A4 | 95.8      | **98.3**  | **99.2**   |
| W2A8 | 93.3      | **99.2**  | **97.5**   |
| W2A6 | 93.3      | **97.5**  | **98.3**   |
| W2A4 | 87.5      | 95.8      | **97.5**   |
| W2A2 | 17.5      | 7.5       | 20.8       |


Key findings:

1. **Grow-to-compensate holds.** M and L retain **≥97% CL with 2-bit weights** (L@W2A6 = 98.3, L@W2A4 = 97.5, M@W2A6 = 97.5). A large 2-bit model (L@W2A4 = 97.5) outperforms a small 4-bit model (S@W4A6 = 95.0), confirming that capacity purchased with the bit savings recovers the lost accuracy.
2. **The small student cannot reach the target.** S peaks at 95.8% (W4A4 / W3A4) at any precision; it is too small to absorb quantization noise.
3. **2-bit activations are the hard floor.** W2A2 collapses at every size (S 17.5 / M 7.5 / L 20.8). All viable 2-bit-weight recipes keep activations at ≥4 bits.
4. **The high-accuracy tier is statistically crowded.** With ±2–4 point CIs, most M/L cells ≥97% are tied; the reliable conclusions are the *tiers* (S < M ≈ L) and the W2A2 cliff, not fine intra-tier rankings. Breaking ties would require more episodes or multiple seeds.
5. **Borderline candidates for a longer fine-tune:** M@W2A4 (95.8) and L@W4A6 (96.7) sit just under the line and may cross it with a 3-epoch schedule (target-steps 48,582).

**Recommendation.** For the lowest-precision configuration meeting the target, use **L@W2A6 (98.3%)** for margin or **M@W2A6 (97.5%)** for a smaller footprint; both achieve the 2-bit-weight (LUT-mapping) objective. The prior **S@W4A6** remains a high-precision fallback but does not reach the 2-bit target.

---



## 8. Reproduction

Driver scripts (all under `research/vit_distill/quant/`, launched via `agent_scripts/`):

**Stage 1  : distillation** (M/L; S reuses `siglip_nano_full.pt`):

```bash
python distill_fp32.py --variant tinyvit_m --target-steps 40480 \
    --batch 32 --lr 1e-3 --weight-decay 0.05 --out tinyvit_m_full.pt
```

**Stage 2  : quantize + screen** (per config; PTQ calibration on, recovery loss off):

```bash
python qat.py --variant tinyvit_m --ckpt tinyvit_m_full.pt \
    --weight-bits 2 --act-bits 6 --data libero \
    --target-steps 2000 --downstream-w 0 --calib-batches 16 \
    --stage qat_sweep --out tinyvit_m_w2a6_qat_sweep.pt
```

**Stage 3a  : downstream-consistency fine-tune** (`qatfix_ds`, resume from screen):

```bash
python qat.py --variant tinyvit_m --ckpt tinyvit_m_full.pt \
    --weight-bits 2 --act-bits 6 --downstream-w 1.0 --downstream-max-tokens 256 \
    --resume-quant tinyvit_m_w2a6_qat_sweep.pt --skip-calib \
    --target-steps 2000 --stage qatfix_ds --out tinyvit_m_w2a6_qatfix_ds.pt
```

**Stage 3b  : 2-epoch final fine-tune** (`qatfix_ep`; batch 32 / target 32388 for S,M; batch 16 / target 64776 for L):

```bash
python qat.py --variant tinyvit_m --ckpt tinyvit_m_full.pt \
    --weight-bits 2 --act-bits 6 --downstream-w 1.0 --downstream-max-tokens 256 \
    --lr 5e-5 --lr-schedule constant --batch 32 --aug heavy \
    --resume-quant tinyvit_m_w2a6_qatfix_ds.pt --skip-calib \
    --target-steps 32388 --save-every 1000 --stage qatfix_ep \
    --out tinyvit_m_w2a6_qatfix_ep.pt
```

The multi-arm, self-chaining launcher `agent_scripts/sweep_qat_epoch.sbatch` packs several configurations per node and resubmits until every arm reaches `TARGET`.

**Evaluation  : closed-loop LIBERO** (120 episodes/config):

```bash
python eval_closedloop_quant.py --quant-state tinyvit_m_w2a6_qatfix_ep.pt \
    --suites object,spatial,goal,10 --episodes-per-suite 30
```

Batch-launched per size via `agent_scripts/sweep_finalcl_ep.sh`.

**Full grid of configurations:** 3 sizes × {W4A6, W4A4, W3A6, W3A4, W2A8, W2A6, W2A4, W2A2}.

---



## Appendix  : Consolidated hyperparameters


| stage          | steps                     | optimizer      | lr (schedule)   | batch             | key loss terms            |
| -------------- | ------------------------- | -------------- | --------------- | ----------------- | ------------------------- |
| Distill (fp32) | ~40,480                   | AdamW, wd 0.05 | 1e-3 (cosine)   | 32                | cosine + norm-MSE         |
| PTQ init       | : (16 calib batches)      | :              | :               | 32                | MAX-stat scale init       |
| QAT screen     | 2,000                     | AdamW, wd 0.05 | 2e-4 (cosine)   | 32                | cosine + norm-MSE (DSW 0) |
| qatfix_ds      | 2,000                     | AdamW, wd 0.05 | 2e-4 (cosine)   | 32                | + downstream (DSW 1.0)    |
| qatfix_ep      | 32,388 (S,M) / 64,776 (L) | AdamW, wd 0.05 | 5e-5 (constant) | 32 (S,M) / 16 (L) | + downstream (DSW 1.0)    |


*Figures:* `frontier_heatmap.png` *(24-cell accuracy grid, Wilson CIs),* `cost_smL_prequant.png` *(params & GMACs by size),* `frontier_grid.csv` *(raw values).*