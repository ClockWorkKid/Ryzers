"""Minitron prune-from-teacher for the MolmoAct2 LLM backbone (experiment C).

Implements the Minitron trim step: rank width elements by activation importance
(minitron_importance.py) and *directly slice the teacher's own weights* into a narrower
student -- the crucial "prune-from-teacher" init that (per Muralidharan et al. 2024, Table 1)
starts far ahead of random init and needs only light distillation to heal.

Design choices tuned to MolmoAct2's action-coupling (zero action-expert surgery this step):
  - Keep hidden=2560  -> student.down_proj/up_proj are IDENTITY (residual stream == teacher's).
  - Keep num_kv_heads=8, head_dim=128 -> per-layer KV dim stays 1024, so the frozen
    action-expert context_{k,v}_proj fit natively (no LoRA needed just to run the init).
  - Prune only MLP intermediate neurons and (optionally) query heads (GQA-group-balanced).

Correctness gate (run ALWAYS): first build a FULL-width student (== teacher), copy teacher
weights in, and assert it reproduces the teacher's per-layer KV + flow loss. This catches the
student<->teacher layout subtlety (teacher MLP is silu(2nd)*1st => gate_up halves are SWAPPED).

Usage:
  python minitron_prune.py --importance importance.pt \
      --target-intermediate 7776 --target-heads 32 \
      --out /outputs/.../pruned_step1.pt --report report.json
"""

from __future__ import annotations
import argparse, json, os
import torch


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default=os.environ.get("MM2_CKPT", "allenai/MolmoAct2-LIBERO"))
    ap.add_argument("--repo-id", default="allenai/MolmoAct2-LIBERO-Dataset")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--importance", default=None,
                    help="importance.pt from minitron_importance.py (teacher source). Ignored "
                         "when --from-ckpt is set (importance is re-estimated on the current student).")
    ap.add_argument("--from-ckpt", default=None,
                    help="prune-from-CURRENT: source weights from a healed student ckpt (iterative "
                         "Minitron step) instead of the teacher. Importance re-estimated on it.")
    ap.add_argument("--target-intermediate", type=int, required=True,
                    help="pruned MLP intermediate size (<= teacher 9728)")
    ap.add_argument("--target-heads", type=int, required=True,
                    help="pruned num query heads (<= teacher 32, divisible by num_kv_heads)")
    ap.add_argument("--out", required=True, help="pruned student checkpoint (train_joint --init format)")
    ap.add_argument("--report", default=None, help="JSON report of the init premise-check")
    ap.add_argument("--eval-batches", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--skip-teacher-gate", action="store_true",
                    help="skip the full-width==teacher assertion (NOT recommended)")
    return ap.parse_args()


def teacher_dims(policy):
    tr = policy._backbone().transformer
    a = tr.blocks[0].self_attn
    hidden = tr.ln_f.weight.shape[0]
    interm = tr.blocks[0].mlp.ff_proj.out_features // 2
    return dict(hidden=int(hidden), num_heads=int(a.num_heads),
                num_kv_heads=int(a.num_key_value_heads), head_dim=int(a.head_dim),
                intermediate=int(interm), num_layers=len(tr.blocks))


def make_cfg(dims, num_heads, intermediate):
    import student as S
    return S.StudentConfig(
        teacher_hidden=dims["hidden"], hidden=dims["hidden"], num_layers=dims["num_layers"],
        num_kv_heads=dims["num_kv_heads"], head_dim=dims["head_dim"],
        num_heads=num_heads, intermediate=intermediate)


@torch.no_grad()
def copy_teacher_into_student(policy, student, cfg):
    """Full-width copy (student cfg must equal teacher dims). Identity in/out projections;
    per-layer att_proj->qkv, attn_out->o, q/k_norm, ff_proj->gate_up (HALVES SWAPPED),
    ff_out->down, attn_norm/ff_norm, ln_f->final_norm."""
    tr = policy._backbone().transformer
    dt = student.down_proj.weight.dtype
    H = cfg.hidden
    eye = torch.eye(H, dtype=dt)
    student.down_proj.weight.copy_(eye)
    student.up_proj.weight.copy_(eye)
    student.final_norm.weight.copy_(tr.ln_f.weight.to(dt))
    interm = cfg.intermediate
    for i, blk in enumerate(tr.blocks):
        sl = student.layers[i]
        sl.attn.qkv.weight.copy_(blk.self_attn.att_proj.weight.to(dt))
        sl.attn.o.weight.copy_(blk.self_attn.attn_out.weight.to(dt))
        if sl.attn.c.use_qk_norm:
            sl.attn.q_norm.weight.copy_(blk.self_attn.q_norm.weight.to(dt))
            sl.attn.k_norm.weight.copy_(blk.self_attn.k_norm.weight.to(dt))
        sl.attn_norm.weight.copy_(blk.attn_norm.weight.to(dt))
        sl.ff_norm.weight.copy_(blk.ff_norm.weight.to(dt))
        # teacher ff_proj = [up ; gate]  (fwd: x,gate=chunk; act(gate)*x) ; student wants [gate ; up]
        up_w, gate_w = blk.mlp.ff_proj.weight.to(dt).chunk(2, dim=0)
        sl.mlp.gate_up.weight.copy_(torch.cat([gate_w, up_w], dim=0))
        sl.mlp.down.weight.copy_(blk.mlp.ff_out.weight.to(dt))
    return student


def select_keep(importance, dims, target_heads, target_interm):
    """Return per-layer keep indices. MLP: top-target_interm neurons. Heads: GQA-balanced,
    top-(target_heads/num_kv_heads) query heads WITHIN each kv group (kv heads all kept)."""
    L = dims["num_layers"]; H = dims["num_heads"]; KV = dims["num_kv_heads"]
    rep = H // KV
    rep_new = target_heads // KV
    assert target_heads % KV == 0, f"target_heads {target_heads} must be divisible by num_kv_heads {KV}"
    mlp_imp = importance["mlp_neuron"]     # [L, interm]
    head_imp = importance["q_head"]        # [L, H]
    keep_mlp, keep_head = [], []
    for li in range(L):
        km = torch.topk(mlp_imp[li], target_interm).indices.sort().values
        keep_mlp.append(km)
        kh = []
        for g in range(KV):
            gidx = torch.arange(g * rep, (g + 1) * rep)
            top = torch.topk(head_imp[li][gidx], rep_new).indices.sort().values
            kh.append(gidx[top])
        keep_head.append(torch.cat(kh))    # group-major order preserved
    return keep_mlp, keep_head


@torch.no_grad()
def slice_to_pruned(full, keep_mlp, keep_head, dims, pruned_cfg):
    """Slice a validated full-width student into a narrower one by keep indices."""
    import student as S
    pruned = S.StudentTextModel(pruned_cfg)
    dt = full.down_proj.weight.dtype
    pruned = pruned.to(dt)
    hd = dims["head_dim"]; H = dims["num_heads"]; KV = dims["num_kv_heads"]
    qdim_full = H * hd; kvdim = KV * hd
    pruned.down_proj.weight.copy_(full.down_proj.weight)
    pruned.up_proj.weight.copy_(full.up_proj.weight)
    pruned.final_norm.weight.copy_(full.final_norm.weight)
    for li in range(dims["num_layers"]):
        fl, pl = full.layers[li], pruned.layers[li]
        pl.attn_norm.weight.copy_(fl.attn_norm.weight)
        pl.ff_norm.weight.copy_(fl.ff_norm.weight)
        if pl.attn.c.use_qk_norm:
            pl.attn.q_norm.weight.copy_(fl.attn.q_norm.weight)
            pl.attn.k_norm.weight.copy_(fl.attn.k_norm.weight)
        # --- attention: keep query-head rows in qkv (q block) + all k,v; keep o cols ---
        kh = keep_head[li]
        q_rows = torch.cat([torch.arange(h * hd, (h + 1) * hd) for h in kh.tolist()])
        w = fl.attn.qkv.weight                     # [qdim_full + 2*kvdim, hidden]
        q_w = w[:qdim_full][q_rows]
        k_w = w[qdim_full:qdim_full + kvdim]
        v_w = w[qdim_full + kvdim:]
        pl.attn.qkv.weight.copy_(torch.cat([q_w, k_w, v_w], dim=0))
        pl.attn.o.weight.copy_(fl.attn.o.weight[:, q_rows])   # o: [hidden, qdim_full]
        # --- MLP: keep neuron rows in gate & up halves + cols in down ---
        km = keep_mlp[li]
        interm_full = fl.mlp.down.weight.shape[1]
        gu = fl.mlp.gate_up.weight                 # [2*interm_full, hidden] = [gate ; up]
        gate_w = gu[:interm_full][km]
        up_w = gu[interm_full:][km]
        pl.mlp.gate_up.weight.copy_(torch.cat([gate_w, up_w], dim=0))
        pl.mlp.down.weight.copy_(fl.mlp.down.weight[:, km])
    return pruned


@torch.no_grad()
def kv_cosine_report(policy, caps, transformer):
    """Per-layer mean KV cosine between the current transformer and stored teacher KV."""
    import torch.nn.functional as F
    perlayer_k, perlayer_v = None, None
    n = 0
    for cap in caps:
        out = transformer(inputs_embeds=cap["inputs_embeds"], attention_mask=cap["attn_bias"],
                          position_ids=cap["positions"], collect_layer_kv_states=True,
                          use_cache=False)
        skv = out.past_key_values
        tkv = cap["teacher_kv"]
        L = len(tkv)
        if perlayer_k is None:
            perlayer_k = [0.0] * L; perlayer_v = [0.0] * L
        for li in range(L):
            tk, tv = tkv[li]
            sk, sv = skv[li]
            ck = F.cosine_similarity(sk.double().flatten(1), tk.double().flatten(1), dim=1).mean()
            cv = F.cosine_similarity(sv.double().flatten(1), tv.double().flatten(1), dim=1).mean()
            perlayer_k[li] += float(ck); perlayer_v[li] += float(cv)
        n += 1
    perlayer_k = [x / n for x in perlayer_k]; perlayer_v = [x / n for x in perlayer_v]
    return perlayer_k, perlayer_v


@torch.no_grad()
def estimate_importance_student(student, caps, num_heads, head_dim):
    """Activation importance (batch=L2, seq=mean) estimated ON a StudentTextModel (iterative
    prune-from-current). Hooks layers[i].mlp.down (input = intermediate neurons) and
    layers[i].attn.o (input = per-head attn output). Reuses the captured teacher inputs_embeds."""
    L = len(student.layers)
    mlp_sq = [None] * L; head_sq = [None] * L
    handles = []

    def mlp_hook(li):
        def h(mod, inp):
            x = inp[0].abs().double()
            m = x.mean(dim=(0, 1))                       # seq/batch mean magnitude [interm]
            mlp_sq[li] = m ** 2 if mlp_sq[li] is None else mlp_sq[li] + m ** 2
        return h

    def attn_hook(li):
        def h(mod, inp):
            x = inp[0]; B, S, _ = x.shape
            ph = x.view(B, S, num_heads, head_dim).double().norm(dim=-1).mean(dim=(0, 1))  # [heads]
            head_sq[li] = ph ** 2 if head_sq[li] is None else head_sq[li] + ph ** 2
        return h

    for li, lyr in enumerate(student.layers):
        handles.append(lyr.mlp.down.register_forward_pre_hook(mlp_hook(li)))
        handles.append(lyr.attn.o.register_forward_pre_hook(attn_hook(li)))
    for cap in caps:
        student(inputs_embeds=cap["inputs_embeds"], attention_mask=cap["attn_bias"],
                position_ids=cap["positions"], collect_layer_kv_states=True, use_cache=False)
    for h in handles:
        h.remove()
    return {"mlp_neuron": torch.stack([torch.sqrt(x) for x in mlp_sq]).float().cpu(),
            "q_head": torch.stack([torch.sqrt(x) for x in head_sq]).float().cpu()}


def main():
    args = get_args()
    device = torch.device(args.device)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    import data as D
    import train_joint as TJ
    import joint_patch as JP

    class A:
        repo_id = args.repo_id; revision = args.revision; teacher = args.teacher
        batch = args.batch; device = args.device; num_workers = args.num_workers

    policy = TJ.build_policy(A, device)
    policy.eval()
    dims = teacher_dims(policy)
    print(f"[minitron-prune] teacher dims: {dims}", flush=True)

    # fixed eval batches with pinned flow t/noise + stored ORIGINAL-teacher KV (the anchor +
    # eval reference; unchanged across the whole curriculum)
    full_iter = D.iter_full_batches(A, device)
    nft = max(1, int(policy.config.num_flow_timesteps))
    heldout = TJ.make_heldout(full_iter, args.eval_batches, nft)
    caps = []
    for item in heldout:
        cap = D.capture_teacher(policy.model, item["batch"], collect_kv=True)
        caps.append({"inputs_embeds": cap["inputs_embeds"], "attn_bias": cap["attn_bias"],
                     "positions": cap["positions"], "teacher_kv": cap["teacher"]["kv"]})
    teacher_flow = TJ.eval_flow(policy, heldout)
    print(f"[minitron-prune] teacher flow baseline = {teacher_flow:.4f}", flush=True)

    import student as S
    dt = next(policy._backbone().transformer.parameters()).dtype
    report = {"teacher_dims": dims, "teacher_flow": teacher_flow}

    if args.from_ckpt:
        # ---- iterative step: prune-from-CURRENT (healed) student --------------------
        ck = torch.load(args.from_ckpt, map_location="cpu", weights_only=False)
        keys = ("hidden", "num_heads", "intermediate", "num_layers", "num_kv_heads",
                "head_dim", "rope_theta", "teacher_hidden", "rms_eps", "use_qk_norm")
        src_cfg = S.make_student_config({k: v for k, v in ck["cfg"].items() if k in keys})
        full = S.StudentTextModel(src_cfg).to(device=device, dtype=dt)
        full.load_state_dict({k: v.to(dt) for k, v in ck["student"].items()}, strict=True)
        src_dims = dict(hidden=src_cfg.hidden, num_heads=src_cfg.num_heads,
                        num_kv_heads=src_cfg.num_kv_heads, head_dim=src_cfg.head_dim,
                        intermediate=src_cfg.intermediate, num_layers=src_cfg.num_layers)
        print(f"[minitron-prune] prune-from-CURRENT source: {src_dims} (ckpt {args.from_ckpt})",
              flush=True)
        imp = estimate_importance_student(full, caps, src_cfg.num_heads, src_cfg.head_dim)
        report["source"] = "current_ckpt"; report["source_dims"] = src_dims
        prune_dims = src_dims
    else:
        # ---- first step: prune-from-TEACHER (with the full-width==teacher gate) ------
        assert args.importance and os.path.exists(args.importance), "need --importance for teacher prune"
        imp = torch.load(args.importance, map_location="cpu")
        assert imp["mlp_neuron"].shape[0] == dims["num_layers"], "importance/teacher layer mismatch"
        full_cfg = make_cfg(dims, dims["num_heads"], dims["intermediate"])
        full = S.StudentTextModel(full_cfg).to(device=device, dtype=dt)
        copy_teacher_into_student(policy, full, full_cfg)
        if not args.skip_teacher_gate:
            gk, gv = kv_cosine_report(policy, caps, full)
            report["fullwidth_kv_cos_k_min"] = round(min(gk), 6)
            report["fullwidth_kv_cos_v_min"] = round(min(gv), 6)
            print(f"[minitron-prune] FULL==TEACHER gate: kv_cos_k_min={min(gk):.6f} "
                  f"kv_cos_v_min={min(gv):.6f} (all 36 layers)", flush=True)
            # bf16 teacher: keys go through qk-norm+RoPE and settle ~0.998-0.999; a real
            # layout/copy bug collapses cosine to ~0. Threshold guards the latter.
            ok = (min(gk) > 0.997 and min(gv) > 0.997)
            report["fullwidth_gate"] = "PASS" if ok else "FAIL"
            if not ok:
                print("[minitron-prune] FULL==TEACHER gate FAILED -- layout/copy bug. Aborting.",
                      flush=True)
                if args.report:
                    json.dump(report, open(args.report, "w"), indent=2)
                raise SystemExit(3)
        report["source"] = "teacher"
        prune_dims = dims

    keep_mlp, keep_head = select_keep(imp, prune_dims, args.target_heads, args.target_intermediate)
    pruned_cfg = make_cfg(prune_dims, args.target_heads, args.target_intermediate)
    pruned = slice_to_pruned(full, keep_mlp, keep_head, prune_dims, pruned_cfg).to(device=device, dtype=dt)
    del full
    n_params = sum(p.numel() for p in pruned.parameters())
    flop_ratio = S.student_flop_ratio(pruned_cfg)
    print(f"[minitron-prune] pruned: heads {prune_dims['num_heads']}->{args.target_heads}  "
          f"interm {prune_dims['intermediate']}->{args.target_intermediate}  "
          f"params={n_params/1e6:.1f}M  approx_reduction={flop_ratio:.2f}x", flush=True)

    # ---- Minitron premise check: pruned-init KV cosine + flow BEFORE healing -----
    pk, pv = kv_cosine_report(policy, caps, pruned)
    JP.attach_student(policy, pruned, lora_rank=8, lora_alpha=16, student_dtype=dt,
                      free_teacher_transformer=False, anchor_weight=0.0)
    pruned_flow = TJ.eval_flow(policy, heldout)
    report.update({
        "target_heads": args.target_heads, "target_intermediate": args.target_intermediate,
        "pruned_params_M": round(n_params / 1e6, 2), "approx_reduction_x": round(flop_ratio, 3),
        "pruned_init_kv_cos_k_min": round(min(pk), 5), "pruned_init_kv_cos_k_mean": round(sum(pk)/len(pk), 5),
        "pruned_init_kv_cos_v_min": round(min(pv), 5), "pruned_init_kv_cos_v_mean": round(sum(pv)/len(pv), 5),
        "pruned_init_flow": round(pruned_flow, 5),
        "flow_gap_vs_teacher": round(pruned_flow - teacher_flow, 5),
    })
    print(f"[minitron-prune] PRUNED-INIT (no healing): kv_cos_k[min={min(pk):.4f} "
          f"mean={sum(pk)/len(pk):.4f}] kv_cos_v[min={min(pv):.4f} mean={sum(pv)/len(pv):.4f}] "
          f"flow={pruned_flow:.4f} (teacher {teacher_flow:.4f}, gap {pruned_flow-teacher_flow:+.4f})",
          flush=True)

    # ---- save pruned checkpoint (train_joint / diag --init format) ---------------
    ckpt = {
        "student": {k: v.detach().to(torch.float32).cpu() for k, v in pruned.state_dict().items()},
        "cfg": pruned_cfg.__dict__, "step": 0, "preset": "minitron",
        "minitron": {"keep_mlp": [k.cpu() for k in keep_mlp],
                     "keep_head": [k.cpu() for k in keep_head],
                     "from_dims": prune_dims, "source": report.get("source"),
                     "from_ckpt": args.from_ckpt,
                     "importance": os.path.abspath(args.importance) if args.importance else None},
    }
    tmp = args.out + ".tmp"
    torch.save(ckpt, tmp); os.replace(tmp, args.out)
    print(f"[minitron-prune] saved pruned student -> {args.out}", flush=True)
    if args.report:
        report["out"] = os.path.abspath(args.out)
        report["perlayer_kv_cos_k"] = [round(x, 4) for x in pk]
        report["perlayer_kv_cos_v"] = [round(x, 4) for x in pv]
        json.dump(report, open(args.report, "w"), indent=2)
        print(f"[minitron-prune] report -> {args.report}", flush=True)


if __name__ == "__main__":
    main()
