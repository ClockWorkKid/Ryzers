"""Module validation for the Workstream-B token-reduction front-end (workspace rule 2).

Loads the trained ROI gate+FastV policy, runs ``capture_teacher_reduced`` on a REAL LIBERO
batch, and checks the reduced fused sequence + teacher targets, then attaches the width-reduced
student and runs the flow-matching loss on the reduced-length context to confirm the student
forward + action-expert cross-attention consume it (masks / position ids / per-layer KV layout).

Run inside the eval container with the FULL roi_overlay bound (gate-inference methods present).
"""

from __future__ import annotations
import argparse, os, traceback
import torch


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roi-ckpt", required=True,
                    help="path to the trained ROI gate pretrained_model dir")
    ap.add_argument("--repo-id", default="allenai/MolmoAct2-LIBERO-Dataset")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--teacher", default=os.environ.get("MM2_CKPT", "allenai/MolmoAct2-LIBERO"))
    ap.add_argument("--preset", default="qwen06w")
    ap.add_argument("--init", default="/outputs/llm_distill/warmstart/qwen06w_init.pt")
    ap.add_argument("--fastv-keep", type=float, default=0.25)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    return ap.parse_args()


def main():
    args = get_args()
    device = torch.device(args.device)
    import data as D
    import student as S
    import joint_patch as JP
    from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy

    print(f"[valB] loading ROI policy from {args.roi_ckpt}", flush=True)
    policy = MolmoAct2Policy.from_pretrained(args.roi_ckpt)
    policy = policy.to(device)
    policy.eval()
    cfg = policy.config
    print(f"[valB] roi_prune_select={getattr(cfg,'roi_prune_select',None)} "
          f"keep_frac={getattr(cfg,'roi_prune_keep_frac',None)} seam={getattr(cfg,'roi_prune_gate_seam',None)} "
          f"fastv_layer={getattr(cfg,'roi_fastv_layer',None)} fastv_keep={getattr(cfg,'roi_fastv_keep_frac',None)}",
          flush=True)

    class Aargs:
        repo_id = args.repo_id; revision = args.revision; teacher = args.teacher
        batch = args.batch; device = args.device; num_workers = 0
    full_iter = D.iter_full_batches(Aargs, device)
    batch = next(full_iter)

    # ---- 1) capture reduced front-end (teacher gate + FastV, prune-before-student) ----
    cap = D.capture_teacher_reduced(policy, batch, fastv_keep=args.fastv_keep, collect_teacher_kv=True)
    red = cap["reduced"]; full = cap["full"]
    S_full = int(full["S"])
    s_new = int(red["inputs_embeds"].shape[1])
    col_idx = red["col_idx"]
    print(f"[valB] full seq S={S_full} -> reduced s_new={s_new} ({100.0*s_new/S_full:.1f}% kept)", flush=True)
    print(f"[valB] reduced inputs_embeds={tuple(red['inputs_embeds'].shape)} "
          f"positions={tuple(red['position_ids'].shape)} mask={tuple(red['attention_mask'].shape)} "
          f"col_idx={tuple(col_idx.shape)}", flush=True)
    assert col_idx.min() >= 0 and col_idx.max() < S_full, "col_idx out of range"
    assert bool((col_idx[:, 1:] > col_idx[:, :-1]).all()), "col_idx must be strictly increasing (sorted keep)"
    tk = cap["teacher"]
    print(f"[valB] teacher kv_full[0].k={tuple(tk['kv_full'][0][0].shape)} "
          f"kv_kept[0].k={tuple(tk['kv_kept'][0][0].shape)} layers={len(tk['kv_full'])}", flush=True)
    assert tk["kv_kept"][0][0].shape[2] == s_new, "kept teacher KV seq must equal reduced length"

    # image-token reduction sanity: how many image tokens survived
    img_id = policy._resolve_image_patch_id()
    ii = batch["input_ids"]
    n_img_full = int((ii == img_id).sum(dim=1)[0].item())
    # kept image tokens = image positions present in col_idx (row 0)
    img_pos = (ii[0] == img_id).nonzero(as_tuple=False).flatten()
    kept0 = set(col_idx[0].tolist())
    n_img_kept = sum(int(p) in kept0 for p in img_pos.tolist())
    print(f"[valB] image tokens: full={n_img_full} kept={n_img_kept} "
          f"({100.0*n_img_kept/max(n_img_full,1):.1f}%) | non-image kept={s_new-n_img_kept}", flush=True)

    # ---- 2) attach the width-reduced student, run flow loss on the REDUCED context ----
    stu, c = S.build_student({"preset": args.preset})
    if args.init and os.path.exists(args.init):
        raw = torch.load(args.init, map_location="cpu", weights_only=False)
        ms, us = stu.load_state_dict(raw["student"], strict=False)
        print(f"[valB] student warmstart missing={len(ms)} unexpected={len(us)}", flush=True)
    groups = JP.attach_student(policy, stu, lora_rank=16, lora_alpha=16,
                               student_dtype=torch.float32, free_teacher_transformer=False,
                               anchor_weight=0.0)
    print(f"[valB] attached student={sum(p.numel() for p in groups['student'])/1e6:.1f}M "
          f"lora={sum(p.numel() for p in groups['lora'])/1e3:.1f}K", flush=True)

    policy.train()
    reduced_mi = {
        "inputs_embeds": red["inputs_embeds"],
        "attention_mask": red["attention_mask"],
        "position_ids": red["position_ids"],
    }
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss, last = policy._compute_flow_matching_loss_joint_per_layer(
            batch=batch, model_inputs=reduced_mi)
    print(f"[valB] REDUCED student flow loss = {float(loss):.4f} (finite={torch.isfinite(loss).item()}) "
          f"last_hidden={tuple(last.shape)}", flush=True)
    assert torch.isfinite(loss).item(), "reduced-path loss is not finite"
    assert last.shape[1] == s_new, f"student last hidden seq {last.shape[1]} != reduced {s_new}"

    # backward smoke: gradients reach the student on the reduced path
    loss.backward()
    g = sum(float(p.grad.norm()) for p in groups["student"] if p.grad is not None)
    gl = sum(float(p.grad.norm()) for p in groups["lora"] if p.grad is not None)
    print(f"[valB] backward OK: student grad-norm-sum={g:.3f} lora grad-norm-sum={gl:.4f}", flush=True)
    assert g > 0, "no gradient reached the student on the reduced path"

    print("[valB] ALL CHECKS PASSED", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
