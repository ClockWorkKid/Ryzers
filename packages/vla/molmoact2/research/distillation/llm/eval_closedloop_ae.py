"""Closed-loop LIBERO launcher with a QUANTIZED action expert swapped in.

Mirrors research/vit_distill/quant/eval_closedloop_quant.py, but instead of the
ViT it rebuilds the Brevitas fake-quant ACTION EXPERT from a QAT blob and loads
it in place. The swap is a lazy patch of ``MolmoAct2Model._require_action_expert``
(the single method the inference flow-denoise loop calls, line ~3054), so it
fires regardless of how lerobot-eval constructs the policy.

Env:
  QUANT_STATE   path to the AE quant blob (weight_bits/act_bits/io_bits/state_dict)
All other CLI args forward verbatim to lerobot-eval.

Requires ``brevitas`` and ``ae_quant`` on PYTHONPATH in the eval container.
"""
from __future__ import annotations

import importlib
import os


def _install_ae_quant_patch():
    state = os.environ.get("QUANT_STATE")
    if not state:
        print("[ae-swap] QUANT_STATE unset -> stock fp action expert", flush=True)
        return
    import torch
    from lerobot.policies.molmoact2.molmoact2_hf_model import modeling_molmoact2 as M
    import ae_quant as AQ

    blob = torch.load(state, map_location="cpu", weights_only=False)
    wbits = int(blob.get("weight_bits", 8))
    abits = int(blob.get("act_bits", 8))
    io_bits = blob.get("io_bits", 8)
    groups = blob.get("groups", None)
    tag = f"W{wbits}A{abits}"
    done = set()
    orig = M.MolmoAct2Model._require_action_expert

    # The static action/depth CUDA-graph capture is incompatible with the
    # Brevitas fake-quant AE: it silently emits degraded actions (W8A8 dropped
    # to ~34% closed-loop, ~6x slower) while the eager path is correct
    # (flow 0.09, actions on par with FP). Force eager whenever we quantize.
    _cg_off = 0
    for cls_name in ("ActionCudaGraphManager", "DepthDecodeCudaGraphManager"):
        cls = getattr(M, cls_name, None)
        if cls is None:
            continue
        for meth in ("can_use_action_flow", "can_use"):
            if hasattr(cls, meth):
                setattr(cls, meth, (lambda *a, **k: False))
                _cg_off += 1
    print(f"[ae-swap] inference CUDA graph force-disabled ({_cg_off} hooks)", flush=True)

    def patched(self):
        ae = orig(self)
        if id(self) not in done:
            dev = next(ae.parameters()).device
            ae.float()
            AQ.quantize_action_expert_(ae, weight_bits=wbits, act_bits=abits,
                                       io_bits=io_bits, groups=groups)
            miss, unexp = ae.load_state_dict(blob["state_dict"], strict=False)
            ae.to(device=dev, dtype=torch.float32)
            for p in ae.parameters():
                p.requires_grad_(False)
            ae.eval()
            done.add(id(self))
            print(f"[ae-swap] {tag} io={io_bits} active on model {id(self)} "
                  f"(load miss={len(miss)} unexp={len(unexp)})", flush=True)
        return ae

    M.MolmoAct2Model._require_action_expert = patched
    print(f"[ae-swap] patched _require_action_expert <- {state} ({tag} io={io_bits})", flush=True)


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
    _install_ae_quant_patch()
    _run_lerobot_eval()
