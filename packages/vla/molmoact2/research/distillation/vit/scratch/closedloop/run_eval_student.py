"""Launcher: swap the MolmoAct2 ViT seam for the distilled student, then run
lerobot-eval unchanged.

Monkey-patches ``MolmoAct2VisionBackbone.encode_image`` BEFORE the policy is
built so every backbone instance produces the student seam. Everything
downstream (2x2 pool, projector, LLM, action head) is untouched, so this is a
pure vision-encoder swap on the pristine MolmoAct2-LIBERO policy.

Env:
  VIT_STUDENT_CKPT    path to hybrid_full.pt (required to activate the swap)
  VIT_STUDENT_MODULE  path to vit_student.py (default /outputs/vit_student.py)

All other CLI args are forwarded verbatim to lerobot-eval.
"""

import importlib.util
import os
import sys


def _load_student_module():
    path = os.environ.get("VIT_STUDENT_MODULE", "/outputs/vit_student.py")
    spec = importlib.util.spec_from_file_location("vit_student", path)
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so @dataclass type resolution can find the module.
    sys.modules["vit_student"] = mod
    spec.loader.exec_module(mod)
    return mod


def _install_student_patch():
    ckpt = os.environ.get("VIT_STUDENT_CKPT")
    if not ckpt:
        print("[student-swap] VIT_STUDENT_CKPT unset -> running stock teacher ViT", flush=True)
        return
    import torch

    from lerobot.policies.molmoact2.molmoact2_hf_model import modeling_molmoact2 as M

    vs = _load_student_module()
    cache = {}

    def encode_image(self, images):
        st = cache.get(id(self))
        if st is None:
            st = vs.build_seam_student()
            raw = torch.load(ckpt, map_location="cpu", weights_only=False)
            missing, unexpected = st.load_state_dict(vs.clean_sd(raw), strict=False)
            if missing or unexpected:
                print(f"[student-swap] load_state_dict missing={list(missing)} "
                      f"unexpected={list(unexpected)}", flush=True)
            st = st.to(device=self.device, dtype=self.dtype).eval()
            for p in st.parameters():
                p.requires_grad_(False)
            n = sum(p.numel() for p in st.parameters())
            cache[id(self)] = st
            print(f"[student-swap] student active on backbone id={id(self)} "
                  f"params={n/1e6:.2f}M dtype={self.dtype} device={self.device}", flush=True)
        return st(images)

    M.MolmoAct2VisionBackbone.encode_image = encode_image
    print(f"[student-swap] patched encode_image <- {ckpt}", flush=True)


def _run_lerobot_eval():
    # Resolve the lerobot-eval console entry point robustly across versions.
    from importlib.metadata import entry_points
    try:
        eps = entry_points(group="console_scripts")
    except TypeError:  # py<3.10 API
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
    import importlib as _il  # noqa
    return target()


if __name__ == "__main__":
    import importlib
    _install_student_patch()
    _run_lerobot_eval()
