"""Launcher: fine-tune the *fake-quant W4A6* student ViT + LoRA (VLM) + action
expert end-to-end on LIBERO, then run lerobot-train unchanged.

This is the quantization-aware sibling of ``run_train_student.py``. Instead of
attaching an fp32 SeamStudent, it attaches the Brevitas fake-quant student that
QAT produced, and keeps it fake-quant (STE) throughout the downstream finetune.
The result is a W4A6-deployable ViT co-adapted with a LoRA'd VLM + finetuned
action expert -- the "student_ft" recipe applied to the quantized encoder.

Design (mirrors qat.py's student construction so the QAT blob loads cleanly):
  1. Build the exact StudentEncoder for the variant from the fp32 base ckpt.
  2. ``quantize_student_`` it to WxAy (fake-quant, float32 numerics + QSDPA).
  3. Load the QAT blob's ``state_dict`` (calibrated scales + trained weights).
  4. Patch ``MolmoAct2VisionBackbone.encode_image`` to route through the attached
     student. The student runs in **float32 with autocast disabled** (brevitas
     fake-quant is exact only in fp32) and its seam is cast back to the policy
     dtype at the boundary; everything downstream stays bf16 mixed-precision.
  5. Patch ``MolmoAct2Policy.__init__`` to attach the (trainable) student AFTER
     the stock LoRA/freeze init. Param names contain "vision" so they land in the
     ViT LR group of ``get_optim_params`` automatically.

Env:
  QUANT_VARIANT   cnn | tinyvit | hybrid            (required)
  FP32_STUDENT    base distilled fp32 ckpt (arch source; overwritten by QAT)  (required)
  QAT_STATE       W4A6 QAT blob (torch.save dict w/ "state_dict")             (required)
  WEIGHT_BITS     default 4
  ACT_BITS        default 6
  QAT_STUDENT_TRAINABLE  "1" (default) trainable fake-quant ViT; "0" freeze it
                         (train only LoRA + action expert on the fixed encoder)

All other CLI args are forwarded verbatim to lerobot-train. Requires the quant/
+ distill packages and brevitas on PYTHONPATH (mount /pkg + /pylibs).
"""

import os
import sys


def _install_trainable_qat_student():
    variant = os.environ.get("QUANT_VARIANT")
    fp32 = os.environ.get("FP32_STUDENT")
    qat = os.environ.get("QAT_STATE")
    if not (variant and fp32 and qat):
        print("[qat-ft] QUANT_VARIANT/FP32_STUDENT/QAT_STATE unset -> training stock teacher ViT",
              flush=True)
        return
    wbits = int(os.environ.get("WEIGHT_BITS", "4"))
    abits = int(os.environ.get("ACT_BITS", "6"))
    trainable = os.environ.get("QAT_STUDENT_TRAINABLE", "1") != "0"

    import torch
    import torch.nn as nn

    from lerobot.policies.molmoact2 import modeling_molmoact2 as POL
    from lerobot.policies.molmoact2.molmoact2_hf_model import modeling_molmoact2 as HFM

    from quant.variants import load_encoder, resolve
    from quant.quant_student import quantize_student_

    variant = resolve(variant)

    class _QuantSeam(nn.Module):
        """Wrap the fake-quant StudentEncoder to accept the backbone's 4D
        [B, crops, N, in_pixels] input (mirrors distill.student.SeamStudent)."""

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

    # 1) encode_image routes through the attached student (fp32, autocast off).
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

    # 2) attach the fake-quant student after the policy is built (post-LoRA).
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
            raise RuntimeError("[qat-ft] could not locate MolmoAct2VisionBackbone")
        if getattr(vb, "student_vit", None) is not None:
            return

        # Build exact arch from fp32 base, fake-quant it, then load QAT weights.
        encoder = load_encoder(variant, fp32).to("cpu", torch.float32)
        encoder = quantize_student_(encoder, weight_bits=wbits, act_bits=abits)
        blob = torch.load(qat, map_location="cpu", weights_only=False)
        state = blob.get("state_dict", blob)
        missing, unexpected = encoder.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"[qat-ft] load QAT state: missing={len(missing)} unexpected={len(unexpected)} "
                  f"(first_missing={list(missing)[:3]} first_unexp={list(unexpected)[:3]})", flush=True)

        st = _QuantSeam(encoder)
        # Keep the fake-quant student in float32 (brevitas STE is exact only in fp32);
        # the seam is cast to the policy dtype at the encode_image boundary.
        st = st.to(dtype=torch.float32)
        dev = next(self.model.parameters()).device
        st = st.to(device=dev)
        for p in st.parameters():
            p.requires_grad_(trainable)
        st.train(mode=trainable)
        vb.student_vit = st  # registers submodule -> shows up in named_parameters()

        n = sum(p.numel() for p in st.parameters())
        n_tr = sum(p.numel() for p in st.parameters() if p.requires_grad)
        print(f"[qat-ft] attached {'TRAINABLE' if trainable else 'FROZEN'} fake-quant "
              f"W{wbits}A{abits} {variant} student: params={n/1e6:.3f}M trainable={n_tr/1e6:.3f}M "
              f"fp32 dev={dev} <- QAT {qat}", flush=True)

    POL.MolmoAct2Policy.__init__ = __init__
    print(f"[qat-ft] patched encode_image + MolmoAct2Policy.__init__ "
          f"(variant={variant} W{wbits}A{abits} trainable={trainable})", flush=True)


if __name__ == "__main__":
    _install_trainable_qat_student()
    from lerobot.scripts.lerobot_train import main
    main()
