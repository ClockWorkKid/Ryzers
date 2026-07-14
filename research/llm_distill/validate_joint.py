"""Module-4a validation: the FUNCTIONAL training path end-to-end on a real batch.

  1) build MolmoAct2Policy (teacher), action_mode='continuous' -> teacher flow loss baseline
     (a real, well-trained model's flow-matching loss scale on this data).
  2) attach the warm-started student (LoRA the action-expert context_{k,v}_proj, free the
     teacher transformer) -> student flow loss (untrained student; expected higher).
  3) loss.backward() -> confirm: loss finite; student params get nonzero grad; action-expert
     LoRA (lora_B) gets nonzero grad; NO grad leaks into the frozen ViT / action-expert base.

This proves gradients flow through student -> KV -> frozen action expert -> flow loss, i.e.
the functional distillation objective is wired correctly, before the full training loop.
"""

from __future__ import annotations
import argparse, os
import torch


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default=os.environ.get("MM2_CKPT", "allenai/MolmoAct2-LIBERO"))
    ap.add_argument("--repo-id", default="allenai/MolmoAct2-LIBERO-Dataset")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--preset", default="qwen06w")
    ap.add_argument("--init", default="/outputs/llm_distill/warmstart/qwen06w_init.pt")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--lora-rank", type=int, default=16)
    return ap.parse_args()


def main():
    args = get_args()
    device = torch.device(args.device)
    ok = True

    import data as D
    import student as S
    import joint_patch as JP
    from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config
    from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy

    class A:  # args for data helpers
        repo_id = args.repo_id; revision = args.revision; teacher = args.teacher
        batch = args.batch; device = args.device; num_workers = args.num_workers
    ds, ds_meta, pre, dcfg = D.build_dataset_and_preprocessor(A)

    cfg = MolmoAct2Config(
        checkpoint_path=args.teacher, chunk_size=10, n_action_steps=10,
        action_mode="continuous", model_dtype="bfloat16", device=args.device,
    )
    # populate input/output features from the dataset (what lerobot-train's factory does)
    from lerobot.configs.types import FeatureType
    try:
        from lerobot.datasets.utils import dataset_to_policy_features
    except Exception:
        from lerobot.policies.factory import dataset_to_policy_features
    feats = dataset_to_policy_features(ds_meta.features)
    cfg.output_features = {k: v for k, v in feats.items() if v.type is FeatureType.ACTION}
    cfg.input_features = {k: v for k, v in feats.items() if v.type is not FeatureType.ACTION}
    print(f"[valj] input_features: {sorted(cfg.input_features)} | "
          f"visual: {[k for k,v in cfg.input_features.items() if v.type is FeatureType.VISUAL]}",
          flush=True)
    print(f"[valj] building policy from {args.teacher} (action_mode=continuous)", flush=True)
    policy = MolmoAct2Policy(cfg)
    policy = policy.to(device)

    batch = next(D.iter_full_batches(A, device))
    print(f"[valj] batch keys: {sorted(k for k in batch if torch.is_tensor(batch[k]))}", flush=True)

    # --- 1) teacher flow-loss baseline (original method, teacher transformer) ---
    policy.eval()
    with torch.no_grad():
        t_loss, t_metrics = policy.forward(batch)
    print(f"[valj] TEACHER continuous flow loss = {float(t_loss):.4f}", flush=True)

    # --- 2) attach warm student + LoRA action expert ---
    stu, c = S.build_student({"preset": args.preset})
    if args.init and os.path.exists(args.init):
        raw = torch.load(args.init, map_location="cpu", weights_only=False)
        ms, us = stu.load_state_dict(raw["student"], strict=False)
        print(f"[valj] warm-start {args.init}: missing={len(ms)} unexpected={len(us)}", flush=True)
    groups = JP.attach_student(policy, stu, lora_rank=args.lora_rank, lora_alpha=16)
    n_stu = sum(p.numel() for p in groups["student"])
    n_lora = sum(p.numel() for p in groups["lora"])
    print(f"[valj] trainable: student={n_stu/1e6:.1f}M lora={n_lora/1e3:.1f}K", flush=True)

    # --- 3) student forward + backward ---
    policy.train()
    s_loss, _ = policy.forward(batch)
    print(f"[valj] STUDENT (untrained) continuous flow loss = {float(s_loss):.4f}", flush=True)
    s_loss.backward()

    checks = []
    checks.append(("teacher flow loss finite", torch.isfinite(t_loss).item()))
    checks.append(("student flow loss finite", torch.isfinite(s_loss).item()))
    g_stu = [p.grad for p in groups["student"] if p.grad is not None]
    stu_grad_ok = len(g_stu) > 0 and any(g.abs().sum().item() > 0 for g in g_stu)
    checks.append(("student params have nonzero grad", stu_grad_ok))
    ae = policy._backbone()._require_action_expert()
    lora_b = [ae.context_k_proj.lora_B, ae.context_v_proj.lora_B]
    lora_grad_ok = all(p.grad is not None for p in lora_b) and any(
        p.grad.abs().sum().item() > 0 for p in lora_b)
    checks.append(("action-expert LoRA (context_k/v) has nonzero grad", lora_grad_ok))
    # frozen leak check: ViT + action-expert base projection should have no grad
    vb = policy._backbone().vision_backbone
    vit_leak = any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in vb.parameters())
    checks.append(("no grad leak into frozen ViT", not vit_leak))
    base_leak = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in ae.context_k_proj.base.parameters())
    checks.append(("no grad leak into frozen action-expert base proj", not base_leak))

    print("\n[valj] ===== RESULTS =====", flush=True)
    for name, passed in checks:
        ok = ok and passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}", flush=True)
    print(f"\n[valj] student/teacher flow-loss ratio = {float(s_loss)/max(float(t_loss),1e-9):.2f}x "
          f"(untrained student expected > teacher)", flush=True)
    print(f"[valj] MODULE-4a {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
