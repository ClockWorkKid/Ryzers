"""Launcher: closed-loop eval of a QAT+LoRA+action-expert co-finetuned checkpoint.

Mirrors ``run_train_qat_student.py`` attach path (fake-quant W4A6 student on the
vision backbone) but freezes the student and loads finetuned weights from
``--policy.path`` (the co-finetune ``pretrained_model`` dir).

Env:
  QUANT_VARIANT   cnn | tinyvit | hybrid            (required)
  FP32_STUDENT    base distilled fp32 ckpt (arch source)  (required)
  QAT_STATE       W4A6 QAT blob for arch bootstrap       (required)
  WEIGHT_BITS     default 4
  ACT_BITS        default 6

All other CLI args forward verbatim to lerobot-eval.
"""

import importlib
import os
import sys


def _install_qat_student_eval():
    variant = os.environ.get("QUANT_VARIANT")
    fp32 = os.environ.get("FP32_STUDENT")
    qat = os.environ.get("QAT_STATE")
    if not (variant and fp32 and qat):
        print("[qat-eval] QUANT_VARIANT/FP32_STUDENT/QAT_STATE unset -> stock teacher ViT",
              flush=True)
        return
    wbits = int(os.environ.get("WEIGHT_BITS", "4"))
    abits = int(os.environ.get("ACT_BITS", "6"))

    import torch
    import torch.nn as nn

    from lerobot.policies.molmoact2 import modeling_molmoact2 as POL
    from lerobot.policies.molmoact2.molmoact2_hf_model import modeling_molmoact2 as HFM

    from quant.quant_student import quantize_student_
    from quant.variants import load_encoder, resolve

    variant = resolve(variant)

    class _QuantSeam(nn.Module):
        def __init__(self, encoder: nn.Module) -> None:
            super().__init__()
            self.encoder = encoder

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            if images.dim() == 3:
                return self.encoder(images)
            b, crops, n, p = images.shape
            x = images.reshape(b * crops, n, p)
            seam = self.encoder(x)
            return seam.reshape(b, crops, n, seam.shape[-1])

    _orig_encode = HFM.MolmoAct2VisionBackbone.encode_image

    def encode_image(self, images):
        st = getattr(self, "student_vit", None)
        if st is None:
            return _orig_encode(self, images)
        out_dtype = images.dtype
        with torch.autocast(device_type="cuda", enabled=False):
            seam = st(images.float())
        return seam.to(out_dtype)

    HFM.MolmoAct2VisionBackbone.encode_image = encode_image

    _orig_init = POL.MolmoAct2Policy.__init__

    def __init__(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        vb = None
        try:
            vb = self._backbone().vision_backbone
        except Exception:
            vb = None
        if vb is None:
            for m in self.model.modules():
                if isinstance(m, HFM.MolmoAct2VisionBackbone):
                    vb = m
                    break
        if vb is None:
            raise RuntimeError("[qat-eval] could not locate MolmoAct2VisionBackbone")
        if getattr(vb, "student_vit", None) is not None:
            return

        encoder = load_encoder(variant, fp32).to("cpu", torch.float32)
        encoder = quantize_student_(encoder, weight_bits=wbits, act_bits=abits)
        blob = torch.load(qat, map_location="cpu", weights_only=False)
        state = blob.get("state_dict", blob)
        missing, unexpected = encoder.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"[qat-eval] bootstrap QAT state: missing={len(missing)} "
                  f"unexpected={len(unexpected)}", flush=True)

        st = _QuantSeam(encoder)
        st = st.to(dtype=torch.float32)
        dev = next(self.model.parameters()).device
        st = st.to(device=dev).eval()
        for p in st.parameters():
            p.requires_grad_(False)
        vb.student_vit = st
        print(f"[qat-eval] attached frozen fake-quant W{wbits}A{abits} {variant} student; "
              f"finetuned weights load from --policy.path", flush=True)

    POL.MolmoAct2Policy.__init__ = __init__
    print(f"[qat-eval] patched encode_image + MolmoAct2Policy.__init__ "
          f"(variant={variant} W{wbits}A{abits})", flush=True)


def _run_lerobot_eval():
    from importlib.metadata import entry_points
    try:
        eps = entry_points(group="console_scripts")
    except TypeError:
        eps = entry_points().get("console_scripts", [])
    for ep in eps:
        if ep.name == "lerobot-eval":
            return ep.load()()
    for modname in ("lerobot.scripts.lerobot_eval", "lerobot.scripts.eval"):
        try:
            mod = importlib.import_module(modname)
            fn = getattr(mod, "main", None) or getattr(mod, "eval_main", None)
            if fn:
                return fn()
        except Exception:
            continue
    raise RuntimeError("could not locate lerobot-eval entry point")


if __name__ == "__main__":
    _install_qat_student_eval()
    _run_lerobot_eval()
