# Action-expert quantization (W8A8 → W2A2)

Low-bit quantization of the MolmoAct2 flow-matching **action expert** (the 36-block
diffusion transformer attached to the frozen backbone), with closed-loop LIBERO as
the metric of record. This is a separate track from the LLM-backbone distillation in
this same package.

Full write-up, tables, heatmap and per-command reproduction: **`results/ae_quant_grid/REPORT.md`**.

## Entry points (this directory)
- `ae_quant.py` — Brevitas fake-quantization of the expert. `quantize_action_expert_(ae, weight_bits, act_bits, io_bits, groups=...)`; `groups` selects which components to quantize (`self_attn,cross_attn,mlp,modulation,io`).
- `ae_qat.py` — PTQ calibration (16 real LIBERO batches) + QAT trainer. `--target-steps 0` = PTQ only.
- `eval_closedloop_ae.py` — loads a saved quantized blob into the policy and runs the LIBERO closed-loop eval (disables the inference CUDA graph, which is incompatible with fake-quant).
- `ae_diag.py` — offline module check (rule 2): compares FP vs quantized expert on held-out batches (flow loss + integrated action MSE) before any full closed-loop run.
- `ae_cl_grouprun.sh` / `ae_cl_inner.sh` — in-container eval driver (one LIBERO suite per GPU).

## Headline result
Uniform quantization collapses even at 8-bit; the sole cause is the **cross-attention**
layers (they read the frozen backbone KV cache). Leaving cross-attention in full
precision (`--groups self_attn,mlp,modulation,io`) and quantizing everything else
recovers full-precision closed-loop success: **W8A8 = 100%** with PTQ only, within a
few points of FP down to **W3A4**, no QAT needed. **2-bit weights are the floor**
(20–35% closed-loop even after QAT).

## Multi-stage reproduction (summary)
1. **Sensitivity** — quantize one group at a time, PTQ only, read held-out flow loss (isolates cross-attention).
2. **Grid PTQ** — sweep W8A8→W2A2 with the cross-attn-FP scheme.
3. **QAT** — 3000 steps on the flow-matching loss for the 2-bit row.
4. **Closed-loop eval** — 4 LIBERO suites, CUDA graphs off.
5. **Artifacts** — `results/ae_quant_grid/make_heatmap.py` regenerates the heatmap.

Group launchers: `slurm/ae_qat_group.sbatch` (PTQ/QAT arms), `slurm/ae_cl_group.sbatch` (closed-loop). See `results/ae_quant_grid/REPORT.md` §4 for exact commands.
