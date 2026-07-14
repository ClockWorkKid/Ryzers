"""Joint functional distillation trainer for the MolmoAct2 LLM student.

Trains the width-reduced student LLM + action-expert LoRA (context_{k,v}_proj) on the stock
flow-matching action loss (action_mode='continuous'), so the student produces KV the frozen
action expert can use. ViT + teacher transformer + rest of action expert stay frozen.

Anti-corruption gate (the exact failure mode from the prior effort): every checkpoint is
(a) reloaded from disk into fresh modules and (b) re-run on a FIXED held-out batch; the run
aborts if the reloaded flow loss doesn't match the in-memory loss. So a saved checkpoint that
"loads but is dead" can never pass silently.

Primary signal is the flow loss trending toward the teacher baseline (~0.73), plus held-out
flow loss (overfit guard). Closed-loop success is the final gate (separate eval job).
"""

from __future__ import annotations
import argparse, json, math, os, time
import torch


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default=os.environ.get("MM2_CKPT", "allenai/MolmoAct2-LIBERO"))
    ap.add_argument("--repo-id", default="allenai/MolmoAct2-LIBERO-Dataset")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--preset", default="qwen06w")
    ap.add_argument("--init", default="/outputs/llm_distill/warmstart/qwen06w_init.pt")
    ap.add_argument("--out-dir", default="/outputs/llm_distill/runs/qwen06w_fn")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--heldout-batches", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    # hybrid: teacher-KV anchoring (representational term on top of flow-matching)
    ap.add_argument("--anchor-weight", type=float, default=0.0,
                    help=">0 keeps the teacher transformer resident and adds all-layer KV anchoring.")
    ap.add_argument("--anchor-mode", choices=["cos", "mse", "both"], default="both")
    ap.add_argument("--anchor-beta", type=float, default=1.0,
                    help="magnitude (normalized-MSE) weight inside the anchor term when mode=both.")
    ap.add_argument("--keep-teacher", action="store_true",
                    help="force keeping the teacher transformer resident even if anchor-weight==0.")
    return ap.parse_args()


def build_policy(args, device):
    import data as D
    from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config
    from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy
    from lerobot.configs.types import FeatureType
    try:
        from lerobot.datasets.utils import dataset_to_policy_features
    except Exception:
        from lerobot.policies.factory import dataset_to_policy_features

    ds, ds_meta, pre, dcfg = D.build_dataset_and_preprocessor(args)
    cfg = MolmoAct2Config(
        checkpoint_path=args.teacher, chunk_size=10, n_action_steps=10,
        action_mode="continuous", model_dtype="bfloat16", device=args.device,
    )
    feats = dataset_to_policy_features(ds_meta.features)
    cfg.output_features = {k: v for k, v in feats.items() if v.type is FeatureType.ACTION}
    cfg.input_features = {k: v for k, v in feats.items() if v.type is not FeatureType.ACTION}
    policy = MolmoAct2Policy(cfg).to(device)
    return policy


def make_heldout(full_iter, n, num_flow_timesteps, seed=1234):
    """Fixed held-out set with PINNED flow timesteps + noise, so eval is deterministic
    (a given model+batch always yields the same flow loss -> the reload gate is exact)."""
    import joint_patch as JP
    ACTION = JP._imports().ACTION
    g = torch.Generator(device="cpu").manual_seed(seed)
    out = []
    for _ in range(n):
        b = next(full_iter)
        a = b[ACTION]
        B, L, Dm = a.shape[0], a.shape[1], a.shape[2]
        t = torch.rand(B, num_flow_timesteps, generator=g)
        noise = torch.randn(B, num_flow_timesteps, L, Dm, generator=g)
        out.append({"batch": b, "t": t.to(a.device), "noise": noise.to(a.device)})
    return out


@torch.no_grad()
def eval_flow(policy, heldout):
    """Deterministic held-out flow loss (uses each item's pinned t/noise via the loss method)."""
    policy.eval()
    tot = 0.0
    for item in heldout:
        b = item["batch"]
        mi = policy._model_inputs(b)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss, _ = policy._compute_flow_matching_loss_joint_per_layer(
                batch=b, model_inputs=mi, timesteps=item["t"], noise=item["noise"])
        tot += float(loss)
    policy.train()
    return tot / max(len(heldout), 1)


def save_checkpoint(policy, c, step, preset, path):
    ae = policy._backbone()._require_action_expert()
    ckpt = {
        "student": {k: v.detach().to(torch.float32).cpu()
                    for k, v in policy.student_llm.state_dict().items()},
        "lora": {
            "ck_A": ae.context_k_proj.lora_A.detach().float().cpu(),
            "ck_B": ae.context_k_proj.lora_B.detach().float().cpu(),
            "cv_A": ae.context_v_proj.lora_A.detach().float().cpu(),
            "cv_B": ae.context_v_proj.lora_B.detach().float().cpu(),
        },
        "lora_scaling": ae.context_k_proj.scaling,
        "cfg": c.__dict__, "step": step, "preset": preset,
    }
    tmp = path + ".tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, path)
    return path


@torch.no_grad()
def verify_saved_checkpoint(policy, path, heldout, inmem_loss, tol=0.02):
    """Reload the saved file into FRESH modules and re-run the held-out batch. Abort if the
    reloaded flow loss disagrees with the in-memory loss (catches silent corruption)."""
    import student as S
    ck = torch.load(path, map_location="cpu", weights_only=False)
    dev = next(policy.student_llm.parameters()).device
    dt = next(policy.student_llm.parameters()).dtype
    fresh, _ = S.build_student({k: v for k, v in ck["cfg"].items()
                                if k in ("hidden", "num_heads", "intermediate", "num_layers",
                                          "num_kv_heads", "head_dim", "rope_theta", "teacher_hidden",
                                          "rms_eps", "use_qk_norm")})
    fresh.load_state_dict(ck["student"], strict=True)
    fresh = fresh.to(device=dev, dtype=dt)
    ae = policy._backbone()._require_action_expert()
    # swap in fresh student + fresh lora, keep originals to restore
    orig_student = policy.student_llm
    orig = {"ckA": ae.context_k_proj.lora_A.data.clone(),
            "ckB": ae.context_k_proj.lora_B.data.clone(),
            "cvA": ae.context_v_proj.lora_A.data.clone(),
            "cvB": ae.context_v_proj.lora_B.data.clone()}
    policy.student_llm = fresh
    ae.context_k_proj.lora_A.data.copy_(ck["lora"]["ck_A"].to(dev, ae.context_k_proj.lora_A.dtype))
    ae.context_k_proj.lora_B.data.copy_(ck["lora"]["ck_B"].to(dev, ae.context_k_proj.lora_B.dtype))
    ae.context_v_proj.lora_A.data.copy_(ck["lora"]["cv_A"].to(dev, ae.context_v_proj.lora_A.dtype))
    ae.context_v_proj.lora_B.data.copy_(ck["lora"]["cv_B"].to(dev, ae.context_v_proj.lora_B.dtype))
    rel = eval_flow(policy, heldout)
    # restore
    policy.student_llm = orig_student
    ae.context_k_proj.lora_A.data.copy_(orig["ckA"]); ae.context_k_proj.lora_B.data.copy_(orig["ckB"])
    ae.context_v_proj.lora_A.data.copy_(orig["cvA"]); ae.context_v_proj.lora_B.data.copy_(orig["cvB"])
    diff = abs(rel - inmem_loss)
    return rel, diff, diff <= max(tol, tol * abs(inmem_loss))


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    logf = open(os.path.join(args.out_dir, "train_log.jsonl"), "a")

    def log(d):
        d["t"] = time.time()
        logf.write(json.dumps(d) + "\n"); logf.flush()
        print("[train] " + " ".join(f"{k}={v}" for k, v in d.items() if k != "t"), flush=True)

    import data as D
    import student as S
    import joint_patch as JP

    policy = build_policy(args, device)

    # teacher baseline (reference target for the flow loss)
    class A:
        repo_id = args.repo_id; revision = args.revision; teacher = args.teacher
        batch = args.batch; device = args.device; num_workers = args.num_workers
    full_iter = D.iter_full_batches(A, device)
    nft = max(1, int(policy.config.num_flow_timesteps))
    heldout = make_heldout(full_iter, args.heldout_batches, nft)
    teacher_base = eval_flow(policy, heldout)
    log({"event": "teacher_baseline", "flow_loss": round(teacher_base, 4)})

    # attach warm student + LoRA (student kept fp32 for stable AdamW; autocast bf16 in fwd)
    stu, c = S.build_student({"preset": args.preset})
    if args.init and os.path.exists(args.init):
        raw = torch.load(args.init, map_location="cpu", weights_only=False)
        ms, us = stu.load_state_dict(raw["student"], strict=False)
        log({"event": "warmstart", "missing": len(ms), "unexpected": len(us)})
    groups = JP.attach_student(policy, stu, lora_rank=args.lora_rank, lora_alpha=16,
                               student_dtype=torch.float32,
                               free_teacher_transformer=not args.keep_teacher,
                               anchor_weight=args.anchor_weight, anchor_mode=args.anchor_mode,
                               anchor_beta=args.anchor_beta)
    anchoring = bool(policy._anchor_cfg.get("enabled"))
    log({"event": "attach", "student_M": round(sum(p.numel() for p in groups["student"]) / 1e6, 1),
         "lora_K": round(sum(p.numel() for p in groups["lora"]) / 1e3, 1),
         "anchor": anchoring, "anchor_w": args.anchor_weight, "anchor_mode": args.anchor_mode,
         "anchor_beta": args.anchor_beta})

    opt = torch.optim.AdamW([
        {"params": groups["student"], "lr": args.lr},
        {"params": groups["lora"], "lr": args.lora_lr},
    ], betas=(0.9, 0.95), weight_decay=args.wd)

    def lr_at(step):
        if step < args.warmup:
            return step / max(args.warmup, 1)
        p = (step - args.warmup) / max(args.steps - args.warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

    policy.train()
    best_heldout = float("inf")
    running = 0.0
    run_flow = 0.0
    run_anchor = 0.0
    t0 = time.time()
    for step in range(1, args.steps + 1):
        scale = lr_at(step)
        for i, g in enumerate(opt.param_groups):
            g["lr"] = (args.lr if i == 0 else args.lora_lr) * scale

        opt.zero_grad(set_to_none=True)
        acc = 0.0
        for _ in range(args.grad_accum):
            batch = next(full_iter)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss, _ = policy.forward(batch)
            (loss / args.grad_accum).backward()
            acc += float(loss) / args.grad_accum
            comp = getattr(policy, "_loss_components", None) or {}
            run_flow += float(comp.get("flow", float(loss))) / args.grad_accum
            run_anchor += float(comp.get("anchor", 0.0)) / args.grad_accum
        gn = torch.nn.utils.clip_grad_norm_(
            [p for g in groups.values() for p in g], args.clip)
        opt.step()
        running += acc

        if step % args.log_every == 0:
            sps = step / (time.time() - t0)
            log({"step": step, "loss": round(running / args.log_every, 4),
                 "flow_loss": round(run_flow / args.log_every, 4),
                 "anchor_loss": round(run_anchor / args.log_every, 4),
                 "grad_norm": round(float(gn), 3), "lr": round(opt.param_groups[0]["lr"], 6),
                 "sps": round(sps, 2), "teacher": round(teacher_base, 4)})
            running = 0.0
            run_flow = 0.0
            run_anchor = 0.0

        if step % args.eval_every == 0:
            ho = eval_flow(policy, heldout)
            log({"step": step, "event": "heldout", "heldout_flow": round(ho, 4),
                 "teacher": round(teacher_base, 4)})
            best_heldout = min(best_heldout, ho)

        if step % args.save_every == 0 or step == args.steps:
            inmem = eval_flow(policy, heldout)
            path = os.path.join(args.out_dir, f"step_{step}.pt")
            save_checkpoint(policy, c, step, args.preset, path)
            rel, diff, okc = verify_saved_checkpoint(policy, path, heldout, inmem)
            log({"step": step, "event": "checkpoint", "path": os.path.basename(path),
                 "inmem_flow": round(inmem, 4), "reload_flow": round(rel, 4),
                 "reload_diff": round(diff, 5), "verify": "PASS" if okc else "FAIL"})
            if not okc:
                log({"event": "ABORT", "reason": "checkpoint reload mismatch (corruption)"})
                raise SystemExit(2)
            # maintain latest.pt symlink-like copy
            save_checkpoint(policy, c, step, args.preset, os.path.join(args.out_dir, "latest.pt"))

    log({"event": "done", "best_heldout_flow": round(best_heldout, 4),
         "teacher": round(teacher_base, 4)})


if __name__ == "__main__":
    main()
