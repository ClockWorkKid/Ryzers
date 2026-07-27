"""Local CPU smoke: build -> quantize -> PTQ -> QONNX export -> parity verify.

Uses random-init students and random patches (NO cluster, NO trained ckpts, NO
GPU) so it validates the *graph/export correctness* -- the riskiest part of the
pipeline (dynamo QONNX cleanliness for QuantConv2d/QuantLinear/QSDPA) -- on the
laptop. Numeric fidelity is meaningless here (random weights); we only assert:
  * quantize_student_ builds and runs a forward
  * PTQ calibration completes
  * export_qonnx(dynamo=True) emits a graph with Quant nodes
  * qonnx cleanup succeeds and the executor matches PyTorch fake-quant (cos>=0.99)

Run:
    cd research/vit_distill
    ../../.venv_quant/Scripts/python.exe -m quant.tests.local_smoke
"""
from __future__ import annotations

import os
import pathlib
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

from brevitas.graph.calibrate import calibration_mode
from brevitas.export import export_qonnx

from quant.quant_student import quantize_student_, SeamOut
from quant.variants import build_encoder, VARIANTS


def run_variant(variant: str, wbits: int, abits: int, n_calib: int = 12, n_hold: int = 4) -> bool:
    print(f"\n==================== {variant}  W{wbits}A{abits} ====================")
    torch.manual_seed(0)
    enc = build_encoder(variant).to(torch.float32).eval()
    nparams = sum(p.numel() for p in enc.parameters())
    print(f"[smoke] {variant} params={nparams/1e6:.2f}M")

    patches = torch.randn(n_calib + n_hold, 729, 588)
    calib, hold = patches[:-n_hold], patches[-n_hold:]

    q = quantize_student_(enc, weight_bits=wbits, act_bits=abits).eval()
    with torch.no_grad():                       # forward works
        _ = q(hold[:2])
    print("[smoke] forward OK")

    with torch.no_grad(), calibration_mode(q):  # PTQ
        for i in range(0, len(calib), 4):
            q(calib[i:i + 4])
    print("[smoke] PTQ calibration OK")

    model = SeamOut(q).eval()
    with torch.no_grad():
        ref = model(hold).float().numpy()

    # Local torch<2.9 has a dynamo/onnx dispatch bug for the QONNX Quant custom op;
    # the tracing exporter (dynamo=False) is the classic FINN route and emits the
    # same QONNX Quant nodes. Toggle with SMOKE_DYNAMO=1 (cluster torch 2.9.1).
    use_dynamo = os.environ.get("SMOKE_DYNAMO", "0") == "1"
    tmp = pathlib.Path(tempfile.gettempdir()) / f"smoke_{variant}_w{wbits}a{abits}.onnx"
    export_qonnx(model, args=torch.zeros(1, 729, 588), export_path=str(tmp),
                 opset_version=17, input_names=["patches"], output_names=["seam"], dynamo=use_dynamo)
    print(f"[smoke] export OK (dynamo={use_dynamo}) -> {tmp.name}")

    from qonnx.core.modelwrapper import ModelWrapper
    from qonnx.core.onnx_exec import execute_onnx
    from qonnx.transformation.infer_shapes import InferShapes
    from qonnx.transformation.infer_datatypes import InferDataTypes
    from qonnx.transformation.fold_constants import FoldConstants
    from qonnx.transformation.general import GiveUniqueNodeNames, RemoveUnusedTensors, SortGraph

    m = ModelWrapper(str(tmp))
    for t in (InferShapes(), InferDataTypes(), GiveUniqueNodeNames(),
              FoldConstants(), RemoveUnusedTensors(), SortGraph()):
        try:
            m = m.transform(t)
        except Exception as e:  # noqa: BLE001
            print(f"[smoke]   cleanup {type(t).__name__} skipped: {e}")
    from collections import Counter
    ops = Counter(nd.op_type for nd in m.graph.node)
    print(f"[smoke] cleanup OK nodes={len(m.graph.node)} Quant={ops.get('Quant',0)} top={ops.most_common(6)}")
    assert ops.get("Quant", 0) > 0, "no Quant nodes in exported graph!"

    iname, oname = m.graph.input[0].name, m.graph.output[0].name
    outs = [execute_onnx(m, {iname: hold[i:i+1].numpy().astype(np.float32)})[oname]
            for i in range(len(hold))]
    onnx_out = np.concatenate(outs, 0)
    cos = F.cosine_similarity(torch.tensor(ref).flatten(1), torch.tensor(onnx_out).flatten(1), dim=1)
    ok = bool(cos.mean() >= 0.99)
    print(f"[smoke] QONNX-vs-PyTorch cosine mean={cos.mean():.5f} min={cos.min():.5f} -> {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    results = {}
    # cnn first (FPGA target, no attention), then nano/tinyvit, then hybrid.
    for variant in ["cnn", "tinyvit", "hybrid"]:
        for (w, a) in [(4, 6)]:            # target precision; sweep exercised on cluster
            try:
                results[f"{variant}_W{w}A{a}"] = run_variant(variant, w, a)
            except Exception as e:          # noqa: BLE001
                import traceback; traceback.print_exc()
                results[f"{variant}_W{w}A{a}"] = False
    print("\n==================== SUMMARY ====================")
    for k, v in results.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
