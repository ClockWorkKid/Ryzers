"""Two-teacher co-distillation of ONE student (LLM + action-expert pair) against BOTH the
MolmoAct2-DROID and MolmoAct2-LIBERO teachers (Track B).

Rationale (capacity test): give the width-reduced 0.6B student broad VLA competence by
distilling the DROID Franka teacher on DROID data AND the LIBERO teacher on LIBERO data into a
SINGLE shared student backbone, then (Track B3) fine-tune on LIBERO to see whether 0.6B width can
approach the full model's ~87%. Both teachers share the Molmo2-ER VLM architecture (hidden 2560,
36 layers, 8 KV heads, head_dim 128), so the student's per-layer KV contract is identical for
both -> one student can be KV-anchored to either teacher. Both teachers pad actions to a 32-dim
space, so ONE shared student action expert serves both embodiments (per-domain normalization is
applied by each domain's preprocessor, so batch actions arrive already normalized).

Design:
  * Build pl (LIBERO policy) + pd (DROID policy). Attach the SAME student module to both.
  * Share ONE action expert (initialized from the LIBERO teacher AE, the downstream target),
    adapted per --adapt-ae {full,lora}; both policies point their backbone.action_expert at it.
  * Each policy keeps its OWN teacher transformer resident (frozen, bf16) as the KV-anchor source
    and its OWN ViT for producing that domain's fused hidden states.
  * Mixed iterator alternates DROID/LIBERO batches (--droid-mix). Only one teacher runs per step,
    so peak activation memory ~ single-teacher training; both teacher WEIGHTS stay resident.
  * Gradients from both domains accumulate into the shared {student LLM, action expert}.

Checkpoint is the coadapt format (plain student + plain action_expert) -> directly consumable by
the existing eval / DAgger / LoRA-finetune paths for Track B3/B4.
"""
from __future__ import annotations
import argparse, json, math, os, random, time
import torch

import data as D
import student as S
import joint_patch as JP
import train_joint as TJ


def get_args():
    ap = argparse.ArgumentParser()
    # teachers / data
    ap.add_argument("--libero-teacher", default="allenai/MolmoAct2-LIBERO")
    ap.add_argument("--libero-repo", default="allenai/MolmoAct2-LIBERO-Dataset")
    ap.add_argument("--droid-teacher", default="allenai/MolmoAct2-DROID")
    ap.add_argument("--droid-repo", default="lerobot/droid_1.0.1")
    ap.add_argument("--droid-norm-tag", default="franka_droid")
    ap.add_argument("--droid-episodes", default="",
                    help="comma-sep DROID episode indices (bounded offline subset). Empty=auto file-0.")
    ap.add_argument("--droid-max-episodes", type=int, default=24)
    ap.add_argument("--video-backend", default="pyav")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--preset", default="qwen06w")
    ap.add_argument("--init", default="/outputs/llm_distill/warmstart/qwen06w_init.pt")
    ap.add_argument("--out-dir", default="/outputs/llm_distill/codistill/qwen06w_dl")
    # mixing / optimization
    ap.add_argument("--droid-mix", type=float, default=0.5, help="fraction of DROID batches")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--ae-lr", type=float, default=None)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--adapt-ae", choices=["full", "lora"], default="full",
                    help="shared student action-expert adaptation during co-distillation.")
    ap.add_argument("--lora-rank", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=64)
    # anchoring (per-domain, each to its own teacher)
    ap.add_argument("--anchor-weight", type=float, default=1.0)
    ap.add_argument("--anchor-mode", choices=["cos", "mse", "both"], default="both")
    ap.add_argument("--anchor-beta", type=float, default=0.1)
    # runtime
    ap.add_argument("--student-dtype", choices=["fp32", "bf16"], default="fp32")
    ap.add_argument("--grad-checkpoint", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--heldout-batches", type=int, default=3)
    ap.add_argument("--resume", default="")
    return ap.parse_args()


def _domain_args(base, *, teacher, repo, norm_tag=None, episodes=None):
    from types import SimpleNamespace
    return SimpleNamespace(
        teacher=teacher, repo_id=repo, revision=base.revision, device=base.device,
        batch=base.batch, num_workers=base.num_workers,
        norm_tag=norm_tag, episodes=episodes, video_backend=base.video_backend,
    )


def _resolve_droid_episodes(args):
    if args.droid_episodes.strip():
        return [int(x) for x in args.droid_episodes.split(",") if x.strip()]
    # auto: episodes fully present in the downloaded file-000 (data + all 3 camera videos)
    import glob, pandas as pd
    snaps = glob.glob("/cache/hub/datasets--lerobot--droid_1.0.1/snapshots/*")
    md = os.path.join(sorted(snaps)[0], "meta")
    epfiles = sorted(glob.glob(os.path.join(md, "episodes", "**", "*.parquet"), recursive=True))
    df = pd.concat([pd.read_parquet(f) for f in epfiles], ignore_index=True)
    cams = ["observation.images.exterior_1_left", "observation.images.exterior_2_left",
            "observation.images.wrist_left"]
    keep = []
    for _, row in df.iterrows():
        dfi = int(row["data/file_index"]) if "data/file_index" in df.columns else None
        vfi = [int(row.get(f"videos/{c}/file_index", -1)) for c in cams]
        if dfi == 0 and all(v == 0 for v in vfi):
            keep.append(int(row["episode_index"]))
        if len(keep) >= args.droid_max_episodes:
            break
    return sorted(set(keep))


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    logf = open(os.path.join(args.out_dir, "train_log.jsonl"), "a")

    def log(d):
        d["t"] = time.time()
        logf.write(json.dumps(d) + "\n"); logf.flush()
        print("[codistill] " + " ".join(f"{k}={v}" for k, v in d.items() if k != "t"), flush=True)

    sdt = torch.float32 if args.student_dtype == "fp32" else torch.bfloat16

    # ---- build both teacher policies -----------------------------------------------------------
    droid_eps = _resolve_droid_episodes(args)
    log({"event": "droid_episodes", "n": len(droid_eps), "sample": droid_eps[:8]})
    a_lib = _domain_args(args, teacher=args.libero_teacher, repo=args.libero_repo)
    a_dro = _domain_args(args, teacher=args.droid_teacher, repo=args.droid_repo,
                         norm_tag=args.droid_norm_tag, episodes=droid_eps)

    log({"event": "build_libero_policy"})
    pl = TJ.build_policy(a_lib, device)
    log({"event": "build_droid_policy"})
    pd = TJ.build_policy(a_dro, device)

    # ---- shared student (warm-start) -----------------------------------------------------------
    _CFG_KEYS = ("hidden", "num_heads", "intermediate", "num_layers", "num_kv_heads",
                 "head_dim", "rope_theta", "teacher_hidden", "rms_eps", "use_qk_norm")
    raw = torch.load(args.init, map_location="cpu", weights_only=False) if args.init and os.path.exists(args.init) else None
    if raw is not None and isinstance(raw.get("cfg"), dict):
        stu, c = S.build_student({k: v for k, v in raw["cfg"].items() if k in _CFG_KEYS})
        miss, unexp = stu.load_state_dict(raw["student"], strict=False)
        log({"event": "student_init_from_cfg", "missing": len(miss), "unexpected": len(unexp)})
    else:
        stu, c = S.build_student({"preset": args.preset})
        log({"event": "student_init_preset", "preset": args.preset})
    stu.grad_checkpoint = bool(args.grad_checkpoint)

    # ---- attach shared student + shared action expert -----------------------------------------
    # LIBERO first: its attach adapts the (shared) action expert, initialized from the LIBERO
    # teacher AE (the downstream target). Anchor ON -> keeps the LIBERO teacher transformer resident.
    groups_l = JP.attach_student(
        pl, stu, adapt_ae=args.adapt_ae, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
        student_dtype=sdt, free_teacher_transformer=False,
        anchor_weight=args.anchor_weight, anchor_mode=args.anchor_mode, anchor_beta=args.anchor_beta)
    shared_ae = pl._backbone()._require_action_expert()

    # DROID: attach the SAME student, DON'T create a second AE (adapt_ae='none'); keep the DROID
    # teacher transformer resident for anchoring; then point DROID's AE at the shared AE.
    groups_d = JP.attach_student(
        pd, stu, adapt_ae="none", student_dtype=sdt, free_teacher_transformer=False,
        anchor_weight=args.anchor_weight, anchor_mode=args.anchor_mode, anchor_beta=args.anchor_beta)
    pd._backbone().action_expert = shared_ae  # share the exact module (weights + grads)

    # trainable param set (student + shared AE / LoRA), de-duplicated by id
    seen, params = set(), []
    for g in (groups_l.get("student", []), groups_l.get("action_expert", []),
              groups_l.get("lora", [])):
        for p in g:
            if id(p) not in seen and p.requires_grad:
                seen.add(id(p)); params.append(p)
    n_student = sum(p.numel() for p in groups_l.get("student", []))
    n_ae = sum(p.numel() for p in groups_l.get("action_expert", []))
    n_lora = sum(p.numel() for p in groups_l.get("lora", []))
    log({"event": "attach", "adapt_ae": args.adapt_ae, "student_M": round(n_student / 1e6, 1),
         "ae_M": round(n_ae / 1e6, 1), "lora_K": round(n_lora / 1e3, 1),
         "n_trainable_tensors": len(params)})

    # ---- optimizer ------------------------------------------------------------------------------
    ae_lr = args.ae_lr if args.ae_lr is not None else args.lr
    pgroups = [{"params": groups_l.get("student", []), "lr": args.lr}]
    if groups_l.get("action_expert"):
        pgroups.append({"params": groups_l["action_expert"], "lr": ae_lr})
    if groups_l.get("lora"):
        pgroups.append({"params": groups_l["lora"], "lr": ae_lr})
    pgroups = [g for g in pgroups if g["params"]]
    base_lrs = [g["lr"] for g in pgroups]
    opt = torch.optim.AdamW(pgroups, betas=(0.9, 0.95), weight_decay=args.wd)

    def lr_scale(step):
        if step < args.warmup:
            return step / max(args.warmup, 1)
        p = (step - args.warmup) / max(args.steps - args.warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

    # ---- data iterators + heldout (LIBERO, comparable to the single-teacher runs) --------------
    lib_iter = D.iter_full_batches(a_lib, device)
    dro_iter = D.iter_full_batches(a_dro, device)
    nft = max(1, int(pl.config.num_flow_timesteps))
    heldout = TJ.make_heldout(lib_iter, args.heldout_batches, nft)

    @torch.no_grad()
    def eval_heldout():
        pl.eval()
        tot = 0.0
        for item in heldout:
            b = item["batch"]; mi = pl._model_inputs(b)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss, _ = pl._compute_flow_matching_loss_joint_per_layer(
                    batch=b, model_inputs=mi, timesteps=item["t"], noise=item["noise"])
            tot += float(loss)
        pl.train()
        return tot / max(len(heldout), 1)

    start_step = 0
    if args.resume and os.path.exists(args.resume):
        rk = torch.load(args.resume, map_location="cpu", weights_only=False)
        stu.load_state_dict(rk["student"], strict=True)
        if rk.get("action_expert") is not None:
            shared_ae.load_state_dict(rk["action_expert"], strict=True)
        if rk.get("optimizer") is not None:
            opt.load_state_dict(rk["optimizer"])
            for st in opt.state.values():
                for k, v in st.items():
                    if torch.is_tensor(v):
                        st[k] = v.to(device)
        start_step = int(rk.get("step", 0))
        log({"event": "resume", "from": os.path.basename(args.resume), "start_step": start_step})

    def save(step, tag):
        ck = {"cfg": c.__dict__, "step": step, "preset": args.preset, "phase": "coadapt",
              "student": {k: v.detach().float().cpu() for k, v in stu.state_dict().items()},
              "action_expert": {k: v.detach().float().cpu() for k, v in shared_ae.state_dict().items()},
              "codistill": {"droid_mix": args.droid_mix, "adapt_ae": args.adapt_ae}}
        if tag == "latest":
            ck["optimizer"] = opt.state_dict()
        path = os.path.join(args.out_dir, f"{tag}.pt" if tag == "latest" else f"step_{step}.pt")
        tmp = path + ".tmp"; torch.save(ck, tmp); os.replace(tmp, path)
        return path

    # ---- training loop --------------------------------------------------------------------------
    rng = random.Random(args.seed)
    pl.train(); pd.train()
    run = {"loss": 0.0, "flow": 0.0, "anchor": 0.0, "nl": 0, "nd": 0}
    t0 = time.time()
    tb_l = eval_heldout()
    log({"event": "libero_teacher_heldout_baseline", "flow": round(tb_l, 4)})

    for step in range(start_step + 1, args.steps + 1):
        sc = lr_scale(step)
        for g, b in zip(opt.param_groups, base_lrs):
            g["lr"] = b * sc
        opt.zero_grad(set_to_none=True)
        acc = 0.0
        for _ in range(args.grad_accum):
            is_droid = rng.random() < args.droid_mix
            policy = pd if is_droid else pl
            batch = next(dro_iter) if is_droid else next(lib_iter)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss, _ = policy.forward(batch)
            (loss / args.grad_accum).backward()
            acc += float(loss) / args.grad_accum
            comp = getattr(policy, "_loss_components", None) or {}
            run["flow"] += float(comp.get("flow", float(loss))) / args.grad_accum
            run["anchor"] += float(comp.get("anchor", 0.0)) / args.grad_accum
            run["nd" if is_droid else "nl"] += 1
        torch.nn.utils.clip_grad_norm_(params, args.clip)
        opt.step()
        run["loss"] += acc

        if step % args.log_every == 0:
            sps = (step - start_step) / (time.time() - t0)
            log({"step": step, "loss": round(run["loss"] / args.log_every, 4),
                 "flow": round(run["flow"] / args.log_every, 4),
                 "anchor": round(run["anchor"] / args.log_every, 4),
                 "droid_frac": round(run["nd"] / max(run["nd"] + run["nl"], 1), 2),
                 "lr": round(opt.param_groups[0]["lr"], 6), "sps": round(sps, 2)})
            run = {"loss": 0.0, "flow": 0.0, "anchor": 0.0, "nl": 0, "nd": 0}

        if step % args.eval_every == 0:
            log({"step": step, "event": "heldout", "libero_flow": round(eval_heldout(), 4)})

        if step % args.save_every == 0 or step == args.steps:
            p = save(step, "step"); save(step, "latest")
            log({"step": step, "event": "checkpoint", "path": os.path.basename(p)})

    log({"event": "done", "steps": args.steps})


if __name__ == "__main__":
    main()
