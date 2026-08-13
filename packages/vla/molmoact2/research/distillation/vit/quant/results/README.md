# ViT quantization — result artifacts

Small result artifacts for the vision-tower quantization campaigns. Full method and
commands are in `../QUANT_REPORT.md`, `../REPRODUCE.md`, and the per-campaign reports below.

| Dir | Campaign | Summary |
|---|---|---|
| `pareto_2bit/` | **Distill → 2-bit grow-to-compensate** (TinyViT S/M/L) | `TRAINING_RECIPE_REPORT.md`: sweep three student sizes × eight W/A precisions; M/L students hold ≥97% closed-loop at **2-bit weights** (L@W2A6 98.3%). Includes the frontier heatmap, plain-QAT sweep, and Strix-Halo latency plots. |
| `qatfix/` | **W4A6 QAT-fix** | `REPORT.md` + `QATFIX_TABLE`: the corrected short cosine-LR LIBERO QAT recovery for cnn/tinyvit. |
| `closedloop/` | **Closed-loop stage table** | `STAGE_TABLE`: fp32 → W8A8 PTQ → W4A6 PTQ → W4A6 QAT closed-loop success per stage. |
| `finn/` | **QONNX → FINN compatibility** | `FIXES_APPLIED.md` / `FORMAT_COMPARE.md`: streamlining/format notes for the exported graph (LayerNorm + Erf/GELU ops). |

Weights are not committed (see `../WEIGHTS.md` + `../download_weights.sh`). The FPGA/KV260
dataflow back-end is kept local and is not part of this package.
