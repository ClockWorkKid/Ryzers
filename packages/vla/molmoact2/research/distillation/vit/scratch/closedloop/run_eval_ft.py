"""Launcher: closed-loop eval of the FINETUNED distilled-full model.

Unlike run_eval_student.py (which swaps in a frozen standalone student on top of
the pristine teacher policy), this attaches a ``student_vit`` submodule to the
vision backbone BEFORE the finetuned checkpoint is loaded, so the checkpoint's
finetuned student + LoRA(VLM) + action-expert weights all load into the matching
architecture. encode_image then routes through that student.

Use with lerobot-eval --policy.path=<finetuned checkpoint pretrained_model dir>.

Env:
  VIT_STUDENT_MODULE  path to vit_student.py (default /outputs/vit_student.py)
  VIT_STUDENT_CKPT    optional placeholder init (overwritten by the checkpoint)

All other CLI args are forwarded verbatim to lerobot-eval.
"""

import importlib
import importlib.util
import os
import sys


def _load_student_module():
    path = os.environ.get("VIT_STUDENT_MODULE", "/outputs/vit_student.py")
    spec = importlib.util.spec_from_file_location("vit_student", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["vit_student"] = mod
    spec.loader.exec_module(mod)
    return mod


def _install_student_arch():
    import torch

    from lerobot.policies.molmoact2 import modeling_molmoact2 as POL
    from lerobot.policies.molmoact2.molmoact2_hf_model import modeling_molmoact2 as HFM

    vs = _load_student_module()
    placeholder = os.environ.get("VIT_STUDENT_CKPT")

    _orig_encode = HFM.MolmoAct2VisionBackbone.encode_image

    def encode_image(self, images):
        st = getattr(self, "student_vit", None)
        if st is None:
            return _orig_encode(self, images)
        return st(images)

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
            raise RuntimeError("[ft-eval] could not locate MolmoAct2VisionBackbone")
        if getattr(vb, "student_vit", None) is not None:
            return
        import json
        skw = json.loads(os.environ.get("VIT_STUDENT_KWARGS", "{}") or "{}")
        st = vs.build_seam_student(**skw)
        if placeholder and os.path.exists(placeholder):
            raw = torch.load(placeholder, map_location="cpu", weights_only=False)
            st.load_state_dict(vs.clean_sd(raw), strict=False)
        dtype = next(self.model.parameters()).dtype
        st = st.to(dtype=dtype).eval()
        for p in st.parameters():
            p.requires_grad_(False)
        vb.student_vit = st  # checkpoint load will overwrite these weights
        print(f"[ft-eval] attached student_vit arch to vision_backbone (dtype={dtype}); "
              f"finetuned weights load from --policy.path", flush=True)

    POL.MolmoAct2Policy.__init__ = __init__
    print("[ft-eval] patched encode_image + MolmoAct2Policy.__init__", flush=True)


def _run_lerobot_eval():
    from importlib.metadata import entry_points
    try:
        eps = entry_points(group="console_scripts")
    except TypeError:
        eps = entry_points().get("console_scripts", [])
    target = None
    for ep in eps:
        if ep.name == "lerobot-eval":
            target = ep.load()
            break
    if target is None:
        for modname in ("lerobot.scripts.lerobot_eval", "lerobot.scripts.eval"):
            try:
                mod = importlib.import_module(modname)
                target = getattr(mod, "main", None) or getattr(mod, "eval_main", None)
                if target:
                    break
            except Exception:
                continue
    if target is None:
        raise RuntimeError("could not locate lerobot-eval entry point")
    return target()


if __name__ == "__main__":
    _install_student_arch()
    _run_lerobot_eval()
