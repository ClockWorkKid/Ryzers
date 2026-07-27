"""Export a Brevitas-quantised student to QONNX (FINN-compatible).

Rebuilds the quantised student for a variant, loads a calibrated PTQ (or QAT)
state, and emits a QONNX graph via the Brevitas DYNAMO export path
(``export_qonnx(..., dynamo=True)``). The dynamo path needs the attention to use
``QuantScaledDotProductAttention`` (see ``quant_student.QuantAttnBlock``); the
``cnn`` variant has no attention so it exports trivially.

After export, runs the qonnx ModelWrapper cleanup transforms (shape/dtype infer,
constant fold, dead-tensor removal) and writes ``*_clean.onnx``.

--verify runs held-out patches through both the PyTorch fake-quant student and
the exported ONNX (qonnx executor) and reports cosine similarity (target >=0.99).

    python -m quant.export_qonnx_student --variant cnn \
        --quant-state artifacts/vit_distill/quant/cnn_w4a6_ptq.pt \
        --verify --verify-file artifacts/vit_distill/quant/calib_patches.pt
"""
from __future__ import annotations

import argparse
import pathlib
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

from brevitas.export import export_qonnx

from quant.quant_student import quantize_student_, SeamOut
from quant.variants import load_encoder, resolve


def build_quant_student(variant: str, blob: dict) -> tuple[torch.nn.Module, int, int]:
    wbits = blob.get("weight_bits", 4)
    abits = blob.get("act_bits", 6)
    enc = load_encoder(variant, None).to(torch.float32).eval()   # graph only; weights come from blob
    enc = quantize_student_(enc, weight_bits=wbits, act_bits=abits).eval()
    missing, unexpected = enc.load_state_dict(blob["state_dict"], strict=False)
    if missing:
        print(f"[qonnx] {len(missing)} missing keys (first {missing[:3]})")
    if unexpected:
        print(f"[qonnx] {len(unexpected)} unexpected keys (first {unexpected[:3]})")
    return enc.eval(), wbits, abits


def cleanup(path: pathlib.Path) -> pathlib.Path:
    from qonnx.core.modelwrapper import ModelWrapper
    from qonnx.transformation.infer_shapes import InferShapes
    from qonnx.transformation.infer_datatypes import InferDataTypes
    from qonnx.transformation.fold_constants import FoldConstants
    from qonnx.transformation.general import (
        GiveUniqueNodeNames, GiveUniqueParameterTensors, RemoveUnusedTensors, SortGraph,
    )
    m = ModelWrapper(str(path))
    print(f"[qonnx] cleanup: {len(m.graph.node)} nodes before")
    for t in (InferShapes(), InferDataTypes(), GiveUniqueNodeNames(),
              GiveUniqueParameterTensors(), FoldConstants(), RemoveUnusedTensors(), SortGraph()):
        try:
            m = m.transform(t)
        except Exception as e:  # noqa: BLE001
            print(f"[qonnx]   transform {type(t).__name__} skipped: {e}")
    clean = pathlib.Path(str(path).replace(".onnx", "_clean.onnx"))
    m.save(str(clean))
    ops = Counter(n.op_type for n in m.graph.node)
    print(f"[qonnx] cleanup: {len(m.graph.node)} nodes after  Quant={ops.get('Quant', 0)}")
    print(f"[qonnx] top ops: {ops.most_common(12)}")
    print(f"[qonnx] saved cleaned -> {clean}")
    return clean


def verify(student, onnx_path, verify_file, n=8):
    from qonnx.core.modelwrapper import ModelWrapper
    from qonnx.core.onnx_exec import execute_onnx
    if not verify_file or not pathlib.Path(verify_file).exists():
        print("[verify] no verify patches, skipping parity")
        return
    p = torch.load(verify_file, map_location="cpu", weights_only=False)
    p = p if torch.is_tensor(p) else p["patches"]
    frames = p.reshape(-1, p.shape[-2], p.shape[-1]).float()[-n:]
    with torch.no_grad():
        ref = SeamOut(student).eval()(frames).float().cpu().numpy()
    m = ModelWrapper(str(onnx_path))
    iname = m.graph.input[0].name
    oname = m.graph.output[0].name
    outs = [execute_onnx(m, {iname: frames[i:i + 1].numpy().astype(np.float32)})[oname]
            for i in range(len(frames))]
    onnx_out = np.concatenate(outs, 0)
    cos = F.cosine_similarity(torch.tensor(ref).flatten(1), torch.tensor(onnx_out).flatten(1), dim=1)
    print(f"[verify] QONNX-vs-PyTorch cosine mean={cos.mean():.5f} min={cos.min():.5f}"
          f"  -> {'PASS' if cos.mean() >= 0.99 else 'WARN (<0.99)'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="QONNX export for a quantised student")
    ap.add_argument("--variant", default="cnn")
    ap.add_argument("--quant-state", required=True, help="PTQ/QAT blob from calibrate.py / qat.py")
    ap.add_argument("--out", default=None)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--export-mode", default="auto", choices=["auto", "dynamo", "trace"],
                    help="auto: dynamo then trace fallback (default); dynamo/trace: force one")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--verify-file", default=None)
    args = ap.parse_args()

    variant = resolve(args.variant)
    blob = torch.load(args.quant_state, map_location="cpu", weights_only=False)
    student, wbits, abits = build_quant_student(variant, blob)
    model = SeamOut(student).eval()

    dummy = torch.zeros(1, 729, 588, dtype=torch.float32)
    out = pathlib.Path(args.out) if args.out else pathlib.Path(
        args.quant_state).with_name(f"{variant}_w{wbits}a{abits}_qonnx.onnx")
    out.parent.mkdir(parents=True, exist_ok=True)

    def _export(dynamo: bool):
        print(f"[qonnx] export dynamo={dynamo} -> {out} (opset {args.opset}, input {tuple(dummy.shape)})")
        export_qonnx(model, args=dummy, export_path=str(out), opset_version=args.opset,
                     input_names=["patches"], output_names=["seam"], dynamo=dynamo)

    # auto = prefer dynamo (reference/QSDPA parity), fall back to the tracing
    # exporter (classic FINN route) if the torch/onnx dynamo dispatch fails.
    if args.export_mode == "trace":
        _export(False)
    elif args.export_mode == "dynamo":
        _export(True)
    else:
        try:
            _export(True)
        except Exception as e:  # noqa: BLE001
            print(f"[qonnx] dynamo export failed ({type(e).__name__}: {e}); falling back to tracing")
            _export(False)
    print(f"[qonnx] saved {out}")
    clean = cleanup(out)
    if args.verify:
        verify(student, clean, args.verify_file)


if __name__ == "__main__":
    main()
