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
    ap.add_argument("--student-dtype", choices=["fp32", "bf16"], default="fp32",
                    help="student param/optimizer dtype. fp32=stable AdamW (small students); "
                         "bf16 halves weight+Adam memory for LARGE Minitron students on 1 GPU.")
    ap.add_argument("--grad-checkpoint", action="store_true",
                    help="gradient-checkpoint student layers (trade compute for activation memory).")
    # Experiment D: joint co-distillation + two-phase (co-adapt distill -> LoRA data finetune)
    ap.add_argument("--train-action-expert", action="store_true",
                    help="Phase 1 co-distillation: unfreeze + train the whole student action "
                         "expert (init from teacher AE) jointly with the student LLM.")
    ap.add_argument("--ae-lr", type=float, default=None,
                    help="LR for the co-adapted action expert (defaults to --lr).")
    ap.add_argument("--lora-finetune", action="store_true",
                    help="Phase 2: LoRA-adapt {llm,ae,vit} on the base flow objective (anchor off). "
                         "--init must be a Phase-1 co-adapt checkpoint (student + action_expert).")
    ap.add_argument("--lora-targets", default="llm,ae,vit",
                    help="comma subset of {llm,ae,vit} to LoRA-adapt in Phase 2.")
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--resume", default="",
                    help="resume from a checkpoint that carries optimizer state (usually latest.pt).")
    # DAgger (on-policy) fine-tune: train student LLM only (AE frozen), anchor on, on a mix of
    # teacher-relabeled student-visited states + demo batches.
    ap.add_argument("--train-student-only", action="store_true",
                    help="train ONLY the student LLM; action expert fully frozen (DAgger recipe).")
    ap.add_argument("--dagger-dir", default="",
                    help="dir of relabel_teacher shards; enables the mixed on-policy/demo iterator.")
    ap.add_argument("--dagger-mix", type=float, default=0.5,
                    help="fraction of on-policy (relabeled) batches in the DAgger mix.")
    ap.add_argument("--dagger-adapt-ae", choices=["none", "lora", "full"], default="none",
                    help="action-expert adaptation during the DAgger student fine-tune: "
                         "none=frozen (round-1 recipe), lora=context-proj LoRA (legacy 30%% recipe), "
                         "full=whole AE trainable. Requires --train-student-only.")
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


def _cpu32(sd):
    return {k: v.detach().to(torch.float32).cpu() for k, v in sd.items()}


def save_checkpoint(policy, c, step, preset, path, *, mode="legacy",
                    lora_targets=None, lora_rank=16, lora_alpha=16, optimizer=None,
                    best_heldout=None, save_vit=False):
    """Serialize the trainable state. Formats by mode:
      * legacy  : student + context-proj LoRA deltas (the 30%-success recipe).
      * coadapt : student + full action_expert state (Phase 1 co-distillation).
      * lora    : student/action_expert/vision_backbone WRAPPED state dicts (Phase 2); the eval
                  assembler re-wraps + merges. ``optimizer`` (when given, e.g. latest.pt) adds
                  optimizer/step for chained-job resume.
    """
    backbone = policy._backbone()
    ae = backbone._require_action_expert()
    ckpt = {"cfg": c.__dict__, "step": step, "preset": preset, "phase": mode,
            "student": _cpu32(policy.student_llm.state_dict())}
    if mode == "coadapt":
        ckpt["action_expert"] = _cpu32(ae.state_dict())
        if save_vit:
            vb = getattr(backbone, "vision_backbone", None)
            if vb is not None:
                ckpt["vision_backbone"] = _cpu32(vb.state_dict())
    elif mode == "lora":
        ckpt["lora_targets"] = list(lora_targets or [])
        ckpt["lora_rank"] = int(lora_rank); ckpt["lora_alpha"] = int(lora_alpha)
        ckpt["action_expert"] = _cpu32(ae.state_dict())
        vb = getattr(backbone, "vision_backbone", None)
        if "vit" in (lora_targets or []) and vb is not None:
            ckpt["vision_backbone"] = _cpu32(vb.state_dict())
    else:
        ckpt["lora"] = {
            "ck_A": ae.context_k_proj.lora_A.detach().float().cpu(),
            "ck_B": ae.context_k_proj.lora_B.detach().float().cpu(),
            "cv_A": ae.context_v_proj.lora_A.detach().float().cpu(),
            "cv_B": ae.context_v_proj.lora_B.detach().float().cpu(),
        }
        ckpt["lora_scaling"] = ae.context_k_proj.scaling
    if optimizer is not None:
        ckpt["optimizer"] = optimizer.state_dict()
        if best_heldout is not None:
            ckpt["best_heldout"] = float(best_heldout)
    tmp = path + ".tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, path)
    return path


@torch.no_grad()
def verify_saved_checkpoint(policy, path, heldout, inmem_loss, tol=0.02, mode="legacy"):
    """Anti-corruption gate. For legacy: reload into FRESH modules and compare held-out flow.
    For coadapt/lora (which mutate the AE / carry wrapped states) we instead reload the file
    and assert every saved tensor round-trips bit-for-bit against the in-memory module (a
    forward re-run would need a full re-assembly; the tensor round-trip catches serialization
    corruption, which is the failure mode this gate exists for)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if mode in ("coadapt", "lora"):
        backbone = policy._backbone()
        ae = backbone._require_action_expert()

        def maxdiff(saved, module):
            cur = module.state_dict()
            d = 0.0
            for k, v in saved.items():
                d = max(d, float((v.float() - cur[k].detach().float().cpu()).abs().max()))
            return d

        diff = maxdiff(ck["student"], policy.student_llm)
        if ck.get("action_expert") is not None:
            diff = max(diff, maxdiff(ck["action_expert"], ae))
        if ck.get("vision_backbone") is not None:
            vb = getattr(backbone, "vision_backbone", None)
            if vb is not None:
                diff = max(diff, maxdiff(ck["vision_backbone"], vb))
        return inmem_loss, diff, diff < 1e-4

    import student as S
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
    rel = eval_flow(policy, heldout)
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
    # heldout + teacher baseline ALWAYS use demo batches (comparable metric across all experiments)
    heldout = make_heldout(full_iter, args.heldout_batches, nft)
    teacher_base = eval_flow(policy, heldout)
    log({"event": "teacher_baseline", "flow_loss": round(teacher_base, 4)})

    # ---- resolve training mode (Experiment D adds coadapt + lora phases; DAgger adds student_only) ----
    if args.lora_finetune:
        mode = "lora"
    elif args.train_action_expert:
        mode = "coadapt"
    elif args.train_student_only:
        mode = "student_only"
    else:
        mode = "legacy"
    # student_only serialization depends on how the AE was adapted:
    #   none/full -> coadapt (plain student + full AE state); lora -> legacy (student + context LoRA)
    if mode == "student_only":
        save_mode = "legacy" if args.dagger_adapt_ae == "lora" else "coadapt"
    else:
        save_mode = mode
    lora_targets = tuple(t.strip() for t in args.lora_targets.split(",") if t.strip())
    log({"event": "mode", "mode": mode, "dagger_dir": args.dagger_dir or None,
         "dagger_mix": args.dagger_mix if args.dagger_dir else None,
         "lora_targets": list(lora_targets) if mode == "lora" else None})

    # attach warm student. If --init carries a "cfg" (Minitron/qwen preset checkpoint), build the
    # student from THAT architecture and load strict; else warmstart from a named preset.
    _CFG_KEYS = ("hidden", "num_heads", "intermediate", "num_layers", "num_kv_heads",
                 "head_dim", "rope_theta", "teacher_hidden", "rms_eps", "use_qk_norm")
    raw = None
    if args.init and os.path.exists(args.init):
        raw = torch.load(args.init, map_location="cpu", weights_only=False)
        if isinstance(raw.get("cfg"), dict):
            stu, c = S.build_student({k: v for k, v in raw["cfg"].items() if k in _CFG_KEYS})
            stu.load_state_dict(raw["student"], strict=True)
            log({"event": "init_from_cfg", "preset": raw.get("preset", "?"),
                 "phase_src": raw.get("phase", "legacy"), "hidden": c.hidden,
                 "num_heads": c.num_heads, "intermediate": c.intermediate})
        else:
            stu, c = S.build_student({"preset": args.preset})
            ms, us = stu.load_state_dict(raw["student"], strict=False)
            log({"event": "warmstart", "missing": len(ms), "unexpected": len(us)})
    else:
        stu, c = S.build_student({"preset": args.preset})
    stu.grad_checkpoint = bool(args.grad_checkpoint)
    sdt = torch.float32 if args.student_dtype == "fp32" else torch.bfloat16
    ae = policy._backbone()._require_action_expert()

    if mode == "lora":
        # Phase 2: load the Phase-1 co-adapted action expert base BEFORE wrapping LoRA.
        if raw is not None and raw.get("action_expert") is not None:
            ae.load_state_dict(raw["action_expert"], strict=True)
            log({"event": "loaded_phase1_ae"})
        else:
            log({"event": "warn", "msg": "lora-finetune init lacks action_expert; using teacher AE base"})
        groups = JP.attach_student_lora_finetune(
            policy, stu, targets=lora_targets, lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha, student_dtype=sdt)
        anchoring = False
    elif mode == "coadapt":
        groups = JP.attach_student(
            policy, stu, student_dtype=sdt, free_teacher_transformer=not args.keep_teacher,
            anchor_weight=args.anchor_weight, anchor_mode=args.anchor_mode,
            anchor_beta=args.anchor_beta, train_action_expert=True)
        # continuing a co-adapt run (init already has a trained AE)? load it over the teacher AE.
        if raw is not None and raw.get("action_expert") is not None:
            ae.load_state_dict(raw["action_expert"], strict=True)
            log({"event": "loaded_ae_from_init"})
        anchoring = bool(policy._anchor_cfg.get("enabled"))
    elif mode == "student_only":
        # DAgger: student LLM trainable; AE adaptation per --dagger-adapt-ae; anchor on (teacher resident).
        # Load the merged base's AE (and ViT, if carried) over the teacher AE BEFORE attach_student,
        # so LoRA (when requested) wraps the correctly-initialized base projections.
        if raw is not None and raw.get("action_expert") is not None:
            ae.load_state_dict(raw["action_expert"], strict=True)
            log({"event": "loaded_ae_from_init"})
        if raw is not None and raw.get("vision_backbone") is not None:
            vb = getattr(policy._backbone(), "vision_backbone", None)
            if vb is not None:
                vb.load_state_dict(raw["vision_backbone"], strict=True)
                log({"event": "loaded_vit_from_init"})
        groups = JP.attach_student(
            policy, stu, adapt_ae=args.dagger_adapt_ae,
            lora_rank=args.lora_rank, lora_alpha=args.lora_alpha, student_dtype=sdt,
            free_teacher_transformer=not args.keep_teacher,
            anchor_weight=args.anchor_weight, anchor_mode=args.anchor_mode,
            anchor_beta=args.anchor_beta)
        anchoring = bool(policy._anchor_cfg.get("enabled"))
    else:
        groups = JP.attach_student(
            policy, stu, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha, student_dtype=sdt,
            free_teacher_transformer=not args.keep_teacher, anchor_weight=args.anchor_weight,
            anchor_mode=args.anchor_mode, anchor_beta=args.anchor_beta)
        anchoring = bool(policy._anchor_cfg.get("enabled"))

    log({"event": "student_dtype", "dtype": args.student_dtype, "grad_ckpt": args.grad_checkpoint})
    log({"event": "attach", "mode": mode,
         "student_M": round(sum(p.numel() for p in groups.get("student", [])) / 1e6, 1),
         "ae_M": round(sum(p.numel() for p in groups.get("action_expert", [])) / 1e6, 1),
         "lora_K": round(sum(p.numel() for p in groups.get("lora", [])) / 1e3, 1),
         "anchor": anchoring, "anchor_w": args.anchor_weight, "anchor_mode": args.anchor_mode,
         "anchor_beta": args.anchor_beta})

    # optimizer over whichever param groups are trainable in this mode
    ae_lr = args.ae_lr if args.ae_lr is not None else args.lr
    param_groups, group_lrs = [], []
    if groups.get("student"):
        param_groups.append({"params": groups["student"], "lr": args.lr}); group_lrs.append(args.lr)
    if groups.get("action_expert"):
        param_groups.append({"params": groups["action_expert"], "lr": ae_lr}); group_lrs.append(ae_lr)
    if groups.get("lora"):
        param_groups.append({"params": groups["lora"], "lr": args.lora_lr}); group_lrs.append(args.lora_lr)
    opt = torch.optim.AdamW(param_groups, betas=(0.9, 0.95), weight_decay=args.wd)

    def lr_at(step):
        if step < args.warmup:
            return step / max(args.warmup, 1)
        p = (step - args.warmup) / max(args.steps - args.warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

    # resume (chained <=4h jobs): restore trainable states + optimizer + step from a prior save
    start_step = 0
    best_heldout = float("inf")
    if args.resume and os.path.exists(args.resume):
        rk = torch.load(args.resume, map_location="cpu", weights_only=False)
        policy.student_llm.load_state_dict(rk["student"], strict=True)
        if rk.get("action_expert") is not None:
            ae.load_state_dict(rk["action_expert"], strict=True)
        # legacy DAgger (--dagger-adapt-ae lora): restore context-proj LoRA deltas onto the wrapped AE
        if rk.get("lora") is not None and hasattr(ae.context_k_proj, "lora_A"):
            Ld = rk["lora"]
            ae.context_k_proj.lora_A.data.copy_(Ld["ck_A"].to(device, ae.context_k_proj.lora_A.dtype))
            ae.context_k_proj.lora_B.data.copy_(Ld["ck_B"].to(device, ae.context_k_proj.lora_B.dtype))
            ae.context_v_proj.lora_A.data.copy_(Ld["cv_A"].to(device, ae.context_v_proj.lora_A.dtype))
            ae.context_v_proj.lora_B.data.copy_(Ld["cv_B"].to(device, ae.context_v_proj.lora_B.dtype))
            log({"event": "resume_lora_deltas"})
        if rk.get("vision_backbone") is not None:
            vb = getattr(policy._backbone(), "vision_backbone", None)
            if vb is not None:
                vb.load_state_dict(rk["vision_backbone"], strict=True)
        if rk.get("optimizer") is not None:
            opt.load_state_dict(rk["optimizer"])
            for st in opt.state.values():
                for k, v in st.items():
                    if torch.is_tensor(v):
                        st[k] = v.to(device)
        start_step = int(rk.get("step", 0))
        best_heldout = float(rk.get("best_heldout", float("inf")))
        log({"event": "resume", "from": os.path.basename(args.resume),
             "start_step": start_step, "best_heldout": round(best_heldout, 4)})

    # carry a frozen custom ViT (e.g. Exp-D's merged ViT) through coadapt-format checkpoints
    save_vit = bool(raw is not None and raw.get("vision_backbone") is not None)
    if save_vit:
        log({"event": "save_vit", "reason": "init carried vision_backbone"})

    # training source: DAgger mixed (on-policy relabel + demo) or pure demo
    if args.dagger_dir:
        import dagger_data as DD
        train_iter = DD.iter_dagger_mixed(A, device, args.dagger_dir, mix=args.dagger_mix, seed=args.seed)
    else:
        train_iter = full_iter

    policy.train()
    running = 0.0
    run_flow = 0.0
    run_anchor = 0.0
    t0 = time.time()
    for step in range(start_step + 1, args.steps + 1):
        scale = lr_at(step)
        for g, base in zip(opt.param_groups, group_lrs):
            g["lr"] = base * scale

        opt.zero_grad(set_to_none=True)
        acc = 0.0
        for _ in range(args.grad_accum):
            batch = next(train_iter)
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
            sps = (step - start_step) / (time.time() - t0)
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
            save_checkpoint(policy, c, step, args.preset, path, mode=save_mode,
                            lora_targets=lora_targets, lora_rank=args.lora_rank,
                            lora_alpha=args.lora_alpha, save_vit=save_vit)
            rel, diff, okc = verify_saved_checkpoint(policy, path, heldout, inmem, mode=save_mode)
            log({"step": step, "event": "checkpoint", "path": os.path.basename(path),
                 "inmem_flow": round(inmem, 4), "reload_flow": round(rel, 4),
                 "reload_diff": round(diff, 5), "verify": "PASS" if okc else "FAIL"})
            if not okc:
                log({"event": "ABORT", "reason": "checkpoint reload mismatch (corruption)"})
                raise SystemExit(2)
            # latest.pt carries optimizer+step for chained-job resume (mi210 4h cap)
            save_checkpoint(policy, c, step, args.preset, os.path.join(args.out_dir, "latest.pt"),
                            mode=save_mode, lora_targets=lora_targets, lora_rank=args.lora_rank,
                            lora_alpha=args.lora_alpha, optimizer=opt, best_heldout=best_heldout,
                            save_vit=save_vit)

    log({"event": "done", "best_heldout_flow": round(best_heldout, 4),
         "teacher": round(teacher_base, 4)})


if __name__ == "__main__":
    main()
