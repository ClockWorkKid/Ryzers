"""Merge a base-student checkpoint (any format: legacy / coadapt / lora) into a uniform, plain
init for DAgger fine-tuning.

Uses ``joint_patch.assemble_student_for_eval`` to fold every LoRA delta into plain modules, then
serializes ``{cfg, student, action_expert[, vision_backbone]}`` in the *coadapt* format (plain
state dicts). Both DAgger arms then start from an identical, format-agnostic checkpoint so the
only difference between them is the base they came from.
"""

from __future__ import annotations
import argparse, os
import torch

import diag_closed_loop as CL
import joint_patch as JP
import student as S


def _cpu32(sd):
    return {k: v.detach().to(torch.float32).cpu() for k, v in sd.items()}


def _clean_student_sd(base):
    """Build a FRESH student and load/merge the base's LLM weights into it -- so the saved state
    dict has exactly the StudentTextModel keys (no eval-only carried teacher submodules like
    wte/ln_f/rotary_emb that ``swap_student_for_eval`` attaches)."""
    scfg = {k: v for k, v in base["cfg"].items() if k in JP._CFG_KEYS_EVAL}
    ref, _ = S.build_student(scfg)
    phase = base.get("phase", "legacy")
    if phase == "lora" and "llm" in tuple(base.get("lora_targets", [])):
        r = int(base.get("lora_rank", 16)); a = int(base.get("lora_alpha", 16))
        JP.wrap_linears_with_lora(ref, rank=r, alpha=a)
        ref.load_state_dict(base["student"], strict=True)
        JP.merge_lora_linears_(ref)
    else:
        ref.load_state_dict(base["student"], strict=True)
    return _cpu32(ref.state_dict())


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--teacher", default="allenai/MolmoAct2-LIBERO")
    ap.add_argument("--repo-id", default="allenai/MolmoAct2-LIBERO-Dataset")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--device", default="cuda")
    return ap.parse_args()


def main():
    args = get_args()
    args.student_ckpt = args.base_ckpt  # for CL.build_policy signature parity
    print(f"[merge] building policy + assembling {os.path.basename(args.base_ckpt)}", flush=True)
    policy, cfg, ds_meta = CL.build_policy(args)
    base = torch.load(args.base_ckpt, map_location="cpu", weights_only=False)
    # clean student weights (fresh-build keys only)
    student_sd = _clean_student_sd(base)
    # use the assembler to merge the AE (+ ViT) LoRA in place; ignore its polluted student swap
    policy, _ = JP.assemble_student_for_eval(policy, base)
    ae = policy._backbone()._require_action_expert()

    out = {
        "cfg": base["cfg"],
        "preset": base.get("preset", "qwen06w"),
        "step": int(base.get("step", 0)),
        "phase": "coadapt",                      # plain student + plain AE
        "student": student_sd,
        "action_expert": _cpu32(ae.state_dict()),
        "merged_from": os.path.basename(args.base_ckpt),
        "base_phase": base.get("phase", "legacy"),
    }
    if base.get("vision_backbone") is not None:
        vb = getattr(policy._backbone(), "vision_backbone", None)
        if vb is not None:
            out["vision_backbone"] = _cpu32(vb.state_dict())
            print("[merge] carried merged vision_backbone", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tmp = args.out + ".tmp"
    torch.save(out, tmp)
    os.replace(tmp, args.out)
    print(f"[merge] DONE base_phase={out['base_phase']} step={out['step']} "
          f"-> {args.out} (student {len(out['student'])} tensors, "
          f"ae {len(out['action_expert'])} tensors)", flush=True)


if __name__ == "__main__":
    main()
