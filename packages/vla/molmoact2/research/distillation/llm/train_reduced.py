"""Top-down token-reduction distillation trainer (Workstream B).

Trains a width-reduced student LLM (+ action-expert LoRA) to run on the 25% task-relevant
token stream produced by the trained ROI gate + FastV front-end (prune-before-student). The
front-end (ViT gate keep-50% @ seam6 -> FastV group-drop to 25% image tokens) is FROZEN; the
student LLM consumes the reduced fused sequence and is supervised by the stock flow-matching
loss, with optional teacher-KV anchoring on the kept tokens (Workstream A merged into B).

Base policy = the trained ROI checkpoint (full-width LLM + gate + FastV-tuned action expert),
loaded via from_pretrained so the deployable front-end is reproduced exactly. The full-width
LLM on the 25% stream (~95%) is the ceiling/teacher; this run tests whether the compressed
student can operate in that condensed space.

Anti-corruption gate identical to train_joint: every checkpoint is reloaded into fresh modules
and re-run on a FIXED held-out (reduced) batch; the run aborts on any flow-loss mismatch.
"""

from __future__ import annotations
import argparse, json, math, os, time
import torch


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roi-ckpt", required=True,
                    help="trained ROI gate+FastV pretrained_model dir (the frozen front-end + teacher)")
    ap.add_argument("--repo-id", default="allenai/MolmoAct2-LIBERO-Dataset")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--teacher", default=os.environ.get("MM2_CKPT", "allenai/MolmoAct2-LIBERO"))
    ap.add_argument("--preset", default="qwen06w")
    ap.add_argument("--init", default="/outputs/llm_distill/warmstart/qwen06w_init.pt")
    ap.add_argument("--out-dir", default="/outputs/llm_distill/runs/qwen06w_reduced")
    ap.add_argument("--fastv-keep", type=float, default=0.25)
    ap.add_argument("--steps", type=int, default=6000)
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
    # optional teacher-KV anchoring on the kept tokens (Workstream A merged into B)
    ap.add_argument("--anchor-weight", type=float, default=0.0)
    ap.add_argument("--anchor-mode", choices=["cos", "mse", "both"], default="both")
    ap.add_argument("--anchor-beta", type=float, default=0.1)
    return ap.parse_args()


def make_heldout_reduced(policy, full_iter, n, num_flow_timesteps, fastv_keep, seed=1234):
    """Fixed held-out set of REDUCED model_inputs with pinned flow t/noise. The front-end is
    frozen, so the captured reduced sequence is a stationary function of each held-out batch."""
    import data as D
    import joint_patch as JP
    ACTION = JP._imports().ACTION
    g = torch.Generator(device="cpu").manual_seed(seed)
    out = []
    for _ in range(n):
        b = next(full_iter)
        cap = D.capture_teacher_reduced(policy, b, fastv_keep=fastv_keep, collect_teacher_kv=False)
        red = cap["reduced"]
        mi = {"inputs_embeds": red["inputs_embeds"], "attention_mask": red["attention_mask"],
              "position_ids": red["position_ids"]}
        a = b[ACTION]
        B, L, Dm = a.shape[0], a.shape[1], a.shape[2]
        t = torch.rand(B, num_flow_timesteps, generator=g)
        noise = torch.randn(B, num_flow_timesteps, L, Dm, generator=g)
        out.append({"batch": b, "mi": mi, "t": t.to(a.device), "noise": noise.to(a.device)})
    return out


@torch.no_grad()
def eval_flow_reduced(policy, heldout):
    """Deterministic held-out flow loss on the REDUCED path (pinned t/noise, frozen front-end)."""
    policy.eval()
    tot = 0.0
    for item in heldout:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss, _ = policy._compute_flow_matching_loss_joint_per_layer(
                batch=item["batch"], model_inputs=item["mi"], timesteps=item["t"], noise=item["noise"])
        tot += float(loss)
    policy.train()
    return tot / max(len(heldout), 1)


@torch.no_grad()
def verify_saved_checkpoint_reduced(policy, path, heldout, inmem_loss, tol=0.02):
    """Reload the saved student+LoRA into FRESH modules and re-run the held-out REDUCED batch;
    abort on any flow-loss mismatch (catches silent corruption)."""
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
    rel = eval_flow_reduced(policy, heldout)
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
        print("[trainB] " + " ".join(f"{k}={v}" for k, v in d.items() if k != "t"), flush=True)

    import data as D
    import student as S
    import joint_patch as JP
    from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy

    log({"event": "load_roi", "ckpt": args.roi_ckpt})
    policy = MolmoAct2Policy.from_pretrained(args.roi_ckpt).to(device)
    fastv_keep = float(args.fastv_keep)
    log({"event": "roi_cfg", "select": getattr(policy.config, "roi_prune_select", None),
         "keep_frac": getattr(policy.config, "roi_prune_keep_frac", None),
         "fastv_layer": getattr(policy.config, "roi_fastv_layer", None),
         "fastv_keep": fastv_keep})

    class A:
        repo_id = args.repo_id; revision = args.revision; teacher = args.teacher
        batch = args.batch; device = args.device; num_workers = args.num_workers
    full_iter = D.iter_full_batches(A, device)
    nft = max(1, int(policy.config.num_flow_timesteps))

    # ceiling/teacher baseline: the FULL-width ROI LLM on the reduced 25% stream (~95% deployable)
    heldout = make_heldout_reduced(policy, full_iter, args.heldout_batches, nft, fastv_keep)
    policy.eval()
    teacher_base = eval_flow_reduced(policy, heldout)
    log({"event": "teacher_baseline_reduced", "flow_loss": round(teacher_base, 4)})

    # attach warm student + LoRA; keep the teacher transformer resident (front-end + anchor source)
    stu, c = S.build_student({"preset": args.preset})
    if args.init and os.path.exists(args.init):
        raw = torch.load(args.init, map_location="cpu", weights_only=False)
        ms, us = stu.load_state_dict(raw["student"], strict=False)
        log({"event": "warmstart", "missing": len(ms), "unexpected": len(us)})
    groups = JP.attach_student(policy, stu, lora_rank=args.lora_rank, lora_alpha=16,
                               student_dtype=torch.float32, free_teacher_transformer=False,
                               anchor_weight=args.anchor_weight, anchor_mode=args.anchor_mode,
                               anchor_beta=args.anchor_beta)
    log({"event": "attach", "student_M": round(sum(p.numel() for p in groups["student"]) / 1e6, 1),
         "lora_K": round(sum(p.numel() for p in groups["lora"]) / 1e3, 1),
         "anchor": bool(policy._anchor_cfg.get("enabled")), "anchor_w": args.anchor_weight})

    opt = torch.optim.AdamW([
        {"params": groups["student"], "lr": args.lr},
        {"params": groups["lora"], "lr": args.lora_lr},
    ], betas=(0.9, 0.95), weight_decay=args.wd)

    def lr_at(step):
        if step < args.warmup:
            return step / max(args.warmup, 1)
        p = (step - args.warmup) / max(args.steps - args.warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

    collect_kv = False  # anchoring recomputes teacher KV on the reduced hidden inside the loss
    policy.train()
    best_heldout = float("inf")
    running = run_flow = run_anchor = 0.0
    t0 = time.time()
    for step in range(1, args.steps + 1):
        scale = lr_at(step)
        for i, g in enumerate(opt.param_groups):
            g["lr"] = (args.lr if i == 0 else args.lora_lr) * scale

        opt.zero_grad(set_to_none=True)
        acc = 0.0
        for _ in range(args.grad_accum):
            batch = next(full_iter)
            cap = D.capture_teacher_reduced(policy, batch, fastv_keep=fastv_keep,
                                            collect_teacher_kv=collect_kv)
            red = cap["reduced"]
            mi = {"inputs_embeds": red["inputs_embeds"], "attention_mask": red["attention_mask"],
                  "position_ids": red["position_ids"]}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss, _ = policy._compute_flow_matching_loss_joint_per_layer(
                    batch=batch, model_inputs=mi)
            (loss / args.grad_accum).backward()
            acc += float(loss) / args.grad_accum
            comp = getattr(policy, "_loss_components", None) or {}
            run_flow += float(comp.get("flow", float(loss))) / args.grad_accum
            run_anchor += float(comp.get("anchor", 0.0)) / args.grad_accum
        gn = torch.nn.utils.clip_grad_norm_([p for g in groups.values() for p in g], args.clip)
        opt.step()
        running += acc

        if step % args.log_every == 0:
            sps = step / (time.time() - t0)
            log({"step": step, "loss": round(running / args.log_every, 4),
                 "flow_loss": round(run_flow / args.log_every, 4),
                 "anchor_loss": round(run_anchor / args.log_every, 4),
                 "grad_norm": round(float(gn), 3), "lr": round(opt.param_groups[0]["lr"], 6),
                 "sps": round(sps, 2), "teacher": round(teacher_base, 4)})
            running = run_flow = run_anchor = 0.0

        if step % args.eval_every == 0:
            ho = eval_flow_reduced(policy, heldout)
            log({"step": step, "event": "heldout", "heldout_flow": round(ho, 4),
                 "teacher": round(teacher_base, 4)})
            best_heldout = min(best_heldout, ho)

        if step % args.save_every == 0 or step == args.steps:
            import train_joint as TJ
            inmem = eval_flow_reduced(policy, heldout)
            path = os.path.join(args.out_dir, f"step_{step}.pt")
            TJ.save_checkpoint(policy, c, step, args.preset, path)
            rel, diff, okc = verify_saved_checkpoint_reduced(policy, path, heldout, inmem)
            log({"step": step, "event": "checkpoint", "path": os.path.basename(path),
                 "inmem_flow": round(inmem, 4), "reload_flow": round(rel, 4),
                 "reload_diff": round(diff, 5), "verify": "PASS" if okc else "FAIL"})
            if not okc:
                log({"event": "ABORT", "reason": "checkpoint reload mismatch (corruption)"})
                raise SystemExit(2)
            TJ.save_checkpoint(policy, c, step, args.preset, os.path.join(args.out_dir, "latest.pt"))

    log({"event": "done", "best_heldout_flow": round(best_heldout, 4),
         "teacher": round(teacher_base, 4)})


if __name__ == "__main__":
    main()
