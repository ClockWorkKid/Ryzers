# Reproduce — distill → PTQ → QAT (W4A6), both variants

End-to-end for the two shipped W4A6 winners. Commands are generic (adapt paths,
launcher, and GPU/container specifics to your environment). Weights: see
`WEIGHTS.md` / `download_weights.sh`.

Assumes `distill/` + `quant/` on `PYTHONPATH`, a ROCm/CUDA PyTorch (≥2.8 for
QONNX dynamo export) with `brevitas`, `qonnx`, `onnx`, and the MolmoAct2 policy +
LIBERO frames available. Set `HF_HOME` and offline flags as appropriate.

Run `VAR=cnn` (conv) and `VAR=tinyvit` (nano). fp32 student checkpoints:
`cnn` → `cnn_fpga_full.pt`, `tinyvit` → `siglip_nano_full.pt`.

## 0. (Optional) distill the fp32 students
The 100×-smaller fp32 students are produced by the distillation configs:
```bash
python train.py --config configs/distill_cnn_fpga.yaml      # -> cnn_fpga_full.pt
python train.py --config configs/distill_siglip_nano.yaml   # -> siglip_nano_full.pt
```
Or just fetch them: `bash download_weights.sh`.

## 1. PTQ + Pareto precision sweep
```bash
python -m quant.calibrate --variant "$VAR" \
    --ckpt resource/ckpt/vit_distill/${VAR/cnn/cnn_fpga}${VAR/tinyvit/siglip_nano}_full.pt \
    --source dataset --n 512 --hold 64 \
    --sweep W8A8,W6A6,W4A8,W4A6 \
    --out artifacts/vit_distill/quant
```
Confirms the W4A6 knee. Saves per-precision PTQ blobs (e.g. `*_w8a8_ptq.pt`,
`*_w4a6_ptq.pt`) and a `*_pareto.json`.

## 2. QAT recovery — the winning recipe
Short, cosine-decayed, **LIBERO-only**, starting from the fp32 student + PTQ init:
```bash
python -m quant.qat --variant "$VAR" \
    --ckpt <fp32 student .pt> \
    --weight-bits 4 --act-bits 6 \
    --data libero --aug heavy \
    --lr 2e-4 --lr-schedule cosine --max-steps 2000 \
    --calib-batches 16 --stage qat \
    --out resource/ckpt/vit_distill/quant/${VAR}_w4a6_qat.pt
```
Notes:
- **Short + cosine LR is essential.** Long / constant-LR / DROID-mix re-distill
  regresses closed-loop despite better static seam cosine (see `QUANT_REPORT.md` §4).
- `--downstream-w > 0` enables the (experimental) downstream-consistency term
  (match student vs teacher seam *after* the frozen pool+projector); default off.

## 3. Closed-loop LIBERO eval (the selection metric)
```bash
QUANT_STATE=resource/ckpt/vit_distill/quant/${VAR}_w4a6_qat.pt \
QUANT_VARIANT="$VAR" \
python -m quant.eval_closedloop_quant <lerobot-eval args: suites, --episodes 20, ...>
```
Baselines: `QUANT_STATE=<*_w4a6_ptq.pt>` (PTQ), or `FP32_STUDENT=<fp32 .pt>`
(unquantized student), or neither (stock teacher ViT).

## 4. Clean seam-fidelity check (diagnostic only — do NOT select on this)
```bash
python -m quant.eval_seam --variant "$VAR" \
    --fp32-ckpt <fp32 student .pt> \
    --quant-state resource/ckpt/vit_distill/quant/${VAR}_w4a6_qat.pt --n 256
```

## 5. QONNX export + parity verify
```bash
python -m quant.export_qonnx_student --variant "$VAR" \
    --quant-state resource/ckpt/vit_distill/quant/${VAR}_w4a6_qat.pt \
    --export-mode auto --verify
```
Target parity cosine ≥ 0.99 vs PyTorch fake-quant. The exported graph contains
`Quant` + `LayerNormalization` + `Erf`(GELU) — the ops a dataflow streamlining
pass must absorb.
```
