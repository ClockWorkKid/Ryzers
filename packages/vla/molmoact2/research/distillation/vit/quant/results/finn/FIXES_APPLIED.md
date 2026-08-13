# TinyViT FINN-compatibility fixes — applied & verified

Target: the TinyViT W4A6 closed-loop winner **`tinyvit_w4a6_qatfix_ds.pt`** (98.8 CL).
All work is TinyViT-only (no conv changes). Fixes applied easiest → hardest.

Final artifact: `weights/qonnx/tinyvit_w4a6_qatfix_ds_finn_clean.onnx` (10.9 MB); see `../../WEIGHTS.md` / `../../download_weights.sh` for the hosted blobs.

## Verdict: re-export only — NO retraining
Every issue was an export-stage problem. The learned weights are untouched; the one
code change (QKV split) is a math-preserving reparameterization of the fused qkv.

## What changed (op histogram: shipped → new)
| op | shipped (op17 trace) | new (op20 dynamo, split-qkv) | issue |
|---|---|---|---|
| `Erf` | 5-op erf GELU | **0** | #4 |
| `Gelu` | 0 | **4** (fused) | #4 |
| `Gather` | 12 | **0** | #2 |
| `Split` | 0 | **0** (3 separate q/k/v matmuls) | #2 |
| `MatMul` | 26 (fused qkv) | 34 (separate q/k/v) | #2 |
| `Expand` | 1 (shared zeros) | **0** | #3 |
| node metadata | 0 / N | **216 / 216** | #1 |
| graph inputs | 1 | 1 (`val_*` dead-input pruned) | — |
| opset / ir_version | 17 / 9 | **20 / 10** | #4/#1 |

(An interim fused-qkv+`Split` variant is also available with `--export-mode dynamo` and no
`--split-qkv`; the canonical artifact below uses the reference-matching separate q/k/v.)

## Fix-by-fix
1. **Node metadata (loop-rolling).** Root cause: shipped graph used the TorchScript
   tracer (dynamo silently fell back). Fix: real dynamo export
   (`export_qonnx(..., dynamo=True)`) on the eval SIF (torch 2.10). Now every node
   carries `namespace` + `pkg.torch.onnx.{class_hierarchy,fx_node,name_scopes,stack_trace}`
   — matching the reference.
2. **QKV Gather + transpose.** Root cause: fused-qkv `reshape(b,n,3,H,hd).permute(2,0,3,1,4)`
   + `qkv[0/1/2]` → 3 Gathers/block on a transposed tensor. Fix: `split_fused_qkv_`
   (`quant_student.py`) reparameterizes the fused `qkv` into **separate
   `q_proj`/`k_proj`/`v_proj` QuantLinears** (reference-SigLIP topology) — 3 projection
   matmuls, **no Gather, no Split**. Weight `[3d,d]`/bias split by output-channel thirds;
   per-channel weight scale is stats-recomputed from each slice; the single per-tensor
   input scale is copied to all three. Enabled via `--split-qkv`. **Bit-identical to the
   fused path** (fused-vs-split student max|diff| = 0.0).
3. **Expand on shared zeros.** Root cause: **not** weight-sharing — Brevitas QSDPA emits
   `attn_bias = torch.zeros(L,S)` added (no-op) to scores, shared across blocks via one
   `Expand`. Fix (`export_qonnx_student.py` `elide_zero_attn_bias`): rewire
   `Add(scores, 0) → scores`, drop the Add+Expand. Also unblocks `FoldConstants`.
   Upstream Brevitas left pristine (surgery lives in our cleanup).
4. **GELU.** Root cause: opset 17 has no `Gelu` op → erf decomposition. Fix: export at
   opset 20 → single fused `Gelu` (our `nn.GELU()` is erf/`approximate='none'`).

## Parity (eval SIF)
- (A) fused-vs-split student (QKV reparam): **max|diff| = 0.000e+00 (identical)**
- (B) split PyTorch fake-quant vs exported QONNX: **cosine mean 0.99758, min 0.99121 → PASS**

## Format vs reference (see FORMAT_COMPARE.md)
Matches on all format crux points: ir_version 10, opset domains
`qonnx.custom_op.general=2; default=20`, Quant schema (domain / {narrow,rounding_mode,signed}
/ 4 inputs / ROUND / signed=1 / bit-widths {4,6}), fused Gelu, single `patches` input, and
**neither graph has any Gather or Split** — TinyViT now uses the same separate q/k/v
projection structure as the reference (topology scaled to 4 blocks / dim 240 vs SigLIP's).

## Environment note (reproduction)
The eval SIF had torch 2.10 + brevitas 0.13 but lacked the onnx stack. Installed into a
bind dir (single SIF preserved): `pylibs_onnx` = onnx 1.22, onnxscript 0.7.1,
qonnx 1.0.0, onnxruntime 1.28, onnxoptimizer. Run with
`PYTHONPATH=pylibs_onnx:pylibs_quant:<quantpkg>`.
