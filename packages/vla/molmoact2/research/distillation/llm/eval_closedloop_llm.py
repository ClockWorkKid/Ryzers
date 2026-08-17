"""Closed-loop LIBERO launcher with a QUANTIZED LLM BACKBONE swapped in.

Mirrors ``eval_closedloop_ae.py`` but rebuilds the Brevitas fake-quant backbone
(``MolmoAct2TextModel``) from a PTQ/QAT blob and loads it in place. The swap is a
lazy one-time patch of ``MolmoAct2TextModel.forward``: on the first forward of a
given transformer instance we fake-quantize it (identically to training) and load
the blob, so it fires regardless of how lerobot-eval constructs the policy.

Env:
  QUANT_STATE   path to the backbone quant blob
                (weight_bits/act_bits/io_bits/groups/state_dict)
All other CLI args forward verbatim to lerobot-eval.

Requires ``brevitas``, ``ae_quant`` and ``llm_quant`` on PYTHONPATH in the eval
container.
"""
from __future__ import annotations

import importlib
import os


def _install_backbone_quant_patch():
    state = os.environ.get("QUANT_STATE")
    if not state:
        print("[llm-swap] QUANT_STATE unset -> stock fp backbone", flush=True)
        return
    import torch
    from lerobot.policies.molmoact2.molmoact2_hf_model import modeling_molmoact2 as M
    import llm_quant as LQ

    blob = torch.load(state, map_location="cpu", weights_only=False)
    wbits = int(blob.get("weight_bits", 8))
    abits = int(blob.get("act_bits", 8))
    io_bits = blob.get("io_bits", 8)
    groups = blob.get("groups", None)
    tag = f"W{wbits}A{abits}"

    # The static action/depth CUDA-graph capture is incompatible with the
    # Brevitas fake-quant modules (silently degraded actions + ~6x slowdown);
    # force eager whenever we quantize, exactly as the AE eval does.
    _cg_off = 0
    for cls_name in ("ActionCudaGraphManager", "DepthDecodeCudaGraphManager"):
        cls = getattr(M, cls_name, None)
        if cls is None:
            continue
        for meth in ("can_use_action_flow", "can_use"):
            if hasattr(cls, meth):
                setattr(cls, meth, (lambda *a, **k: False))
                _cg_off += 1
    print(f"[llm-swap] inference CUDA graph force-disabled ({_cg_off} hooks)", flush=True)

    done = set()
    orig_forward = M.MolmoAct2TextModel.forward

    def patched_forward(self, *args, **kwargs):
        if id(self) not in done:
            dev = next(self.parameters()).device
            self.float()
            LQ.quantize_backbone_(self, weight_bits=wbits, act_bits=abits,
                                  io_bits=io_bits, groups=groups)
            miss, unexp = self.load_state_dict(blob["state_dict"], strict=False)
            self.to(device=dev, dtype=torch.float32)
            for p in self.parameters():
                p.requires_grad_(False)
            self.eval()
            done.add(id(self))
            print(f"[llm-swap] {tag} groups={groups} active on transformer {id(self)} "
                  f"(load miss={len(miss)} unexp={len(unexp)})", flush=True)
        return orig_forward(self, *args, **kwargs)

    M.MolmoAct2TextModel.forward = patched_forward
    print(f"[llm-swap] patched MolmoAct2TextModel.forward <- {state} ({tag})", flush=True)


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
        except Exception:  # noqa: BLE001
            continue
    raise RuntimeError("could not locate lerobot-eval entry point")


if __name__ == "__main__":
    _install_backbone_quant_patch()
    _run_lerobot_eval()
