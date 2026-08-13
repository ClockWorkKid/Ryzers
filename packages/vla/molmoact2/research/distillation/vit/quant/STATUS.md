# Quant pipeline — status & gates (Phase 0 findings)

Last updated: 2026-07-24 (UTC-6).

> Historical Phase-0 snapshot. Final results and the shipped W4A6 recipe are in `QUANT_REPORT.md`.

## What is DONE (this session, laptop)
- Full quantization code implemented + **locally validated on CPU** (`.venv_quant`,
  torch 2.8 CPU + brevitas 0.13 + qonnx + onnx): `tests/local_smoke.py` PASSES for
  **cnn / tinyvit / hybrid** at W4A6 (build → quant → PTQ → QONNX export → cleanup →
  parity **cosine ≥ 0.998**). `calibrate.py` + `export_qonnx_student.py` CLIs run
  end-to-end (knee: W8A8≈0.9998 → W4A6≈0.987 on random weights).
- FINN feasibility flags confirmed from the exported `cnn` graph: **`Conv`(16),
  `MatMul`(18), `Quant`(68)** + **`LayerNormalization`(17)** + **`Erf`/GELU(16)** —
  the LN/GELU ops are what streamlining must absorb (blueprint `streamline:` covers this).
- Cluster runners staged under `slurm/`/`scratch/`, quant image layer
  (`Dockerfile.quant`), and the FINN/Brainsmith driver + blueprint kept local
  (not upstreamed).

## GATES / blockers (need cluster resources or user go)
1. **cnn / nano checkpoints must be distilled first.** At the time of this snapshot
   only `hybrid_full.pt` existed (in the cluster checkpoint dir `resource/ckpt/vit_distill/`).
   The `distill_cnn_fpga.yaml` / `distill_siglip_nano.yaml` configs had not been run yet.
   **Prereq:** distill cnn + nano first (~3 h / 4 GPU each). `hybrid` can be quantized now.
2. **Build image if absent.** If the quant container image is not present in the
   image cache, rebuild the base (network=host) then the quant layer.
3. **No FPGA toolchain on the ROCm GPU clusters** (expected): no Vivado/Vitis/FINN/
   Brainsmith on login or PATH. The FPGA stage needs a separate **Vivado + Brainsmith
   host**. The `--stage estimate` FINN path can run on any host with FINN installed
   (no bitstream).
4. **Shared-cluster etiquette:** other jobs may be running on the shared cluster.
   Launch distill/PTQ/QAT within a modest node cap and poll / self-resubmit.

## Recommended next actions (in order)
1. Rebuild image (`quant_build_image.sh`) on a free build node.
2. Distill `cnn` then `nano` (highest priority = cnn, FPGA-first).
3. PTQ sweep → confirm the W4A6 knee on real weights → QAT if it regresses.
4. Closed-loop LIBERO validate each quantized variant (gated on user go for full suites).
5. QONNX export (`--export-mode auto`) → verify cosine ≥ 0.99.
6. On the Vivado/Brainsmith host: streamline → FINN estimate → forge → KV260 bitstream.
