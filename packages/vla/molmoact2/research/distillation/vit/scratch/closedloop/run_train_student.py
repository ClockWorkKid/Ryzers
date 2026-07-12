"""Launcher: fine-tune the distilled student ViT + LoRA (VLM) + action expert
end-to-end on LIBERO, then run lerobot-train unchanged.

This mirrors run_eval_student.py but makes the student a TRAINABLE submodule of
the MolmoAct2 vision backbone instead of a frozen eval-time swap:

  1. Patch ``MolmoAct2VisionBackbone.encode_image`` to route through a
     ``student_vit`` submodule when one is attached (else the stock teacher ViT).
  2. Wrap ``MolmoAct2Policy.__init__`` so that AFTER the original init (which
     applies LoRA + freezes the base model), we attach a trainable SeamStudent
     initialized from ``hybrid_full.pt``. Its parameter names contain "vision",
     so ``get_optim_params`` routes them into the ViT LR group automatically.

Everything else (LoRA on the LLM, fully-trainable action expert, last-N decoder
unfreeze, flow-matching loss, checkpointing) is the stock roi_overlay pipeline
with ROI pruning disabled.

Env:
  VIT_STUDENT_CKPT    path to hybrid_full.pt (required to activate the swap)
  VIT_STUDENT_MODULE  path to vit_student.py (default /outputs/vit_student.py)

All other CLI args are forwarded verbatim to lerobot-train.
"""

import importlib
import importlib.util
import os
import sys


def _load_student_module():
    path = os.environ.get("VIT_STUDENT_MODULE", "/outputs/vit_student.py")
    spec = importlib.util.spec_from_file_location("vit_student", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["vit_student"] = mod  # register before exec for @dataclass resolution
    spec.loader.exec_module(mod)
    return mod


def _install_trainable_student():
    ckpt = os.environ.get("VIT_STUDENT_CKPT")
    if not ckpt:
        print("[student-ft] VIT_STUDENT_CKPT unset -> training stock teacher ViT", flush=True)
        return
    import torch

    from lerobot.policies.molmoact2 import modeling_molmoact2 as POL
    from lerobot.policies.molmoact2.molmoact2_hf_model import modeling_molmoact2 as HFM

    vs = _load_student_module()

    # 1) encode_image routes through the attached student if present.
    _orig_encode = HFM.MolmoAct2VisionBackbone.encode_image

    def encode_image(self, images):
        st = getattr(self, "student_vit", None)
        if st is None:
            return _orig_encode(self, images)
        return st(images)

    HFM.MolmoAct2VisionBackbone.encode_image = encode_image

    # 2) attach a trainable student after the policy is built (post-LoRA).
    _orig_init = POL.MolmoAct2Policy.__init__

    def __init__(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        # locate the vision backbone (handles PeftModel nesting)
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
            raise RuntimeError("[student-ft] could not locate MolmoAct2VisionBackbone")
        if getattr(vb, "student_vit", None) is not None:
            return  # already attached

        import json
        skw = json.loads(os.environ.get("VIT_STUDENT_KWARGS", "{}") or "{}")
        if skw:
            print(f"[student-ft] student kwargs={skw}", flush=True)
        st = vs.build_seam_student(**skw)
        raw = torch.load(ckpt, map_location="cpu", weights_only=False)
        missing, unexpected = st.load_state_dict(vs.clean_sd(raw), strict=False)
        if missing or unexpected:
            print(f"[student-ft] load_state_dict missing={list(missing)} "
                  f"unexpected={list(unexpected)}", flush=True)
        dtype = next(self.model.parameters()).dtype
        st = st.to(dtype=dtype)
        for p in st.parameters():
            p.requires_grad_(True)
        st.train()
        vb.student_vit = st  # registers submodule -> shows up in named_parameters()

        n = sum(p.numel() for p in st.parameters())
        n_tr = sum(p.numel() for p in st.parameters() if p.requires_grad)
        print(f"[student-ft] attached TRAINABLE student to vision_backbone: "
              f"params={n/1e6:.3f}M trainable={n_tr/1e6:.3f}M dtype={dtype} <- {ckpt}",
              flush=True)

    POL.MolmoAct2Policy.__init__ = __init__
    print("[student-ft] patched encode_image + MolmoAct2Policy.__init__", flush=True)


if __name__ == "__main__":
    _install_trainable_student()
    from lerobot.scripts.lerobot_train import main
    main()
