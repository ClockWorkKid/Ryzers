# vit_distill/quant — W4A6 quantization of the distilled students

Extends the distilled MolmoAct2 vision-tower students (`cnn`, `tinyvit`/"nano
siglip", `hybrid`) to a **W4A6** quantized tower (4-bit weights, 6-bit
activations) via Brevitas PTQ + quantization-aware distillation, and exports
verified **QONNX**. The downstream FPGA dataflow back-end (board-specific forge
glue + bitstream build) is kept local and is **not** part of this package.

Re-targets a standard SigLIP quantization recipe from an HF `SiglipVisionModel`
to our custom `distill.student.StudentEncoder`.

## Results & reproduction
- **`QUANT_REPORT.md`** — full dev-flow results (distill → PTQ Pareto → QAT
  recovery → closed-loop → QONNX export) + the key finding (short cosine-LR
  LIBERO QAT is the closed-loop winner; select on closed-loop, not seam cosine).
- **`REPRODUCE.md`** — end-to-end commands.
- **`WEIGHTS.md`** + **`download_weights.sh`** — full dev-flow checkpoints (both
  variants), hosted externally (weights are not committed to git).

## Files
| file | phase | role |
|------|-------|------|
| `quant_student.py` | 1 | Brevitas fake-quant for `StudentEncoder` (`QuantLinear`+`QuantConv2d`, `QuantAttnBlock`/QSDPA) |
| `variants.py`      | – | variant registry (cnn/tinyvit/hybrid) + distilled-ckpt loader |
| `calibrate.py`     | 2 | PTQ calibration + Pareto precision sweep {W8A8,W6A6,W4A8,W4A6} |
| `pareto_plot.py`   | 2 | accuracy-vs-precision knee plot (+ optional closed-loop SR) |
| `qat.py`           | 3 | quantization-aware distillation (seam loss vs frozen teacher) |
| `eval_closedloop_quant.py` | 4 | swap the quantized student into the policy, run `lerobot-eval` |
| `eval_seam.py`     | 4 | clean held-out seam fidelity (cosine / rel-L2) — diagnostic only |
| `export_qonnx_student.py`  | 5 | `export_qonnx` (dynamo→trace auto) + qonnx cleanup + parity verify |
| `tests/local_smoke.py` | – | CPU build→quant→PTQ→export→parity smoke (no GPU/ckpts) |

## Local validation
`tests/local_smoke.py` PASSES for **cnn / tinyvit / hybrid** at W4A6:
build → quantize → PTQ → QONNX export (tracing) → cleanup → parity
**cosine ≥ 0.998** vs PyTorch fake-quant.

> `dynamo=True` export needs torch≥2.8 with a matching onnx/onnxscript. If the
> dynamo dispatcher lacks the QONNX `Quant` custom op, use `--export-mode trace`
> (or `--export-mode auto` to try dynamo first with trace fallback).
