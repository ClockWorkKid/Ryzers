# LLM-backbone quantization (W8A8 → W2A2)

Low-bit quantization of the MolmoAct2 **language-model backbone** (the frozen 36-layer
Qwen3-style transformer whose per-layer key/value cache the action expert reads), with
closed-loop LIBERO as the metric of record. This is the backbone counterpart to the
action-expert track (`AE_QUANT.md`) in this same package.

Full write-up, tables, and per-command reproduction: **`results/ae_quant_grid/REPORT.md`**
(second part, "Low-bit quantization of the MolmoAct2 language-model backbone").

## Entry points (this directory)
- `llm_quant.py` — Brevitas fake-quantization of the backbone. `quantize_backbone_(transformer, weight_bits, act_bits, io_bits, groups=...)`; `groups` selects which components to quantize (`attn_q,attn_kv,attn_sdpa,mlp,io`). The fused QKV projection is split so the query and key/value paths can be quantized independently.
- `llm_qat.py` — PTQ calibration (16 real LIBERO batches) + QAT trainer. `--target-steps 0` = PTQ only. Uses a disjoint episode-level held-out split for checkpoint selection and a seamless (no-recalibration) chunked resume so long QAT runs survive a ~4h scheduler wall-time cap.
- `eval_closedloop_llm.py` — rebuilds the fake-quant backbone from a saved blob and swaps it into the policy, then runs the LIBERO closed-loop eval (disables the inference CUDA graph, which is incompatible with fake-quant).
- `llm_cl_grouprun.sh` / `llm_cl_inner.sh` — in-container eval driver (one LIBERO suite per GPU).
- `data.py` — adds `episode_split()` (deterministic disjoint train/val episodes) and the `episodes_override` plumbing used to build a non-leaky validation iterator.

## Headline result
Uniform quantization collapses even at 8-bit; the sole cause is the **attention math**
(the scaled-dot-product-attention computation), not any weight projection. Leaving the
attention math in bfloat16 (`--groups attn_q,attn_kv,mlp`) and quantizing all three
projection groups recovers full precision: **W8A8 = 100%** closed-loop with PTQ only,
within a few points of the full-precision baseline down to **W4A8 (95%)**. The backbone
is more weight-sensitive than the action expert: **3-bit weights are the failure boundary**.
QAT is double-edged — it recovers the borderline cells (W4A6 35%→75%, W4A4 0%→11%) but
lowers the training loss far below full precision while making closed-loop *worse* on the
cells PTQ already solves (W≥6). Use **PTQ for W≥4A8 and QAT only at the 4-bit margin**.

## Multi-stage reproduction (summary)
1. **Sensitivity** — quantize one group at a time at W8A8, PTQ only, read held-out flow loss (isolates the attention math).
2. **Grid PTQ** — sweep W8A8→W2A2 with the mixed-precision scheme (`attn_q,attn_kv,mlp`).
3. **QAT** — fine-tune on the flow-matching loss (chunked + seamless resume; disjoint held-out selection).
4. **Closed-loop eval** — 4 LIBERO suites, CUDA graphs off.
5. **Artifacts** — `results/llm_quant_grid/make_heatmap.py` regenerates the heatmap.

Group launchers: `slurm/llm_qat_group.sbatch` (grouped PTQ/QAT arms), `slurm/llm_qat_cell_chunk.sbatch` (single chunked+resumable QAT cell), `slurm/llm_cl_group.sbatch` (closed-loop). See `results/ae_quant_grid/REPORT.md` §4 (backbone part) for exact commands.
