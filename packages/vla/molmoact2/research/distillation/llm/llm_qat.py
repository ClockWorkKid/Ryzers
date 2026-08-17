"""PTQ + QAT fine-tuning of the fake-quantized MolmoAct2 LLM BACKBONE.

Quantizes ONLY the LLM backbone transformer (ViT + action expert frozen),
PTQ-calibrates the quant scales on real LIBERO batches, then (optionally) QAT
fine-tunes on the stock flow-matching MSE (``policy.forward(batch)``) to recover
closed-loop accuracy at the target W/A precision.

This mirrors ``ae_qat.py`` with two deliberate differences:
  * The *backbone* is the trainable module (ViT + action expert are frozen).
    Gradients from the flow-matching loss reach the backbone via the per-layer
    K/V that the action expert cross-attends to, so knowledge-insulation is
    turned OFF (the AE recipe turned it ON to keep the backbone frozen).
  * Gradient checkpointing is enabled for the 36 decoder layers to fit
    full-backbone QAT on a single 64 GB GPU.

``--target-steps 0`` runs PTQ only (calibrate + save); ``--target-steps > 0``
runs QAT fine-tuning on the flow-matching loss. This study ran both across the
whole W8A8->W2A2 grid. The mixed-precision scheme quantizes the query, key/value
and MLP projections and leaves the attention math (scaled-dot-product attention)
in bfloat16 -- i.e. ``--groups attn_q,attn_kv,mlp`` (dropping ``attn_sdpa``);
adding ``attn_sdpa`` back is the uniform scheme. QAT recovers the borderline
cells (W4A6, W4A4) but overfits the flow loss on the cells PTQ already solves
(W>=6), so PTQ is preferred at W>=4A8 and QAT only helps at the 4-bit margin.

    # PTQ-only, mixed-precision (attention math kept in bf16):
    python llm_qat.py --weight-bits 4 --act-bits 8 --groups attn_q,attn_kv,mlp \
        --target-steps 0 --out /outputs/llm_quant/mixed_w4a8
    # QAT at the 4-bit margin (mixed-precision, disjoint held-out selection):
    python llm_qat.py --weight-bits 4 --act-bits 6 --groups attn_q,attn_kv,mlp \
        --target-steps 6000 --ae-lr 5e-5 --lr-schedule cosine \
        --val-frac 0.05 --heldout-batches 32 --out /outputs/llm_quant/mixedft_w4a6

Blob format (for eval rebuild):
    {"state_dict": transformer.state_dict(), "weight_bits", "act_bits",
     "io_bits", "groups", "step", "kind": "backbone_qat"}
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", default="allenai/MolmoAct2-LIBERO")
    p.add_argument("--repo-id", default="allenai/MolmoAct2-LIBERO-Dataset")
    p.add_argument("--revision", default="main")
    p.add_argument("--device", default="cuda")
    p.add_argument("--weight-bits", type=int, required=True)
    p.add_argument("--act-bits", type=int, required=True)
    p.add_argument("--io-bits", type=int, default=8,
                   help="parity with ae_qat; io group is a no-op for this backbone")
    p.add_argument("--groups", default=None,
                   help="comma list of quant groups (attn_q,attn_kv,attn_sdpa,mlp,io); "
                        "default=all=uniform. 'attn_q,attn_kv,mlp' (drop attn_sdpa) "
                        "= the mixed-precision scheme (attention math kept in bf16)")
    p.add_argument("--target-steps", type=int, default=0,
                   help="0 = PTQ only (>=3-bit grid); >0 = QAT (2-bit floor)")
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--ae-lr", type=float, default=5e-5)
    p.add_argument("--lr-schedule", default="constant", choices=["constant", "cosine"])
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--calib-batches", type=int, default=16)
    p.add_argument("--quant-dtype", default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--freeze-embed", action="store_true", default=True,
                   help="freeze the token embedding (wte) during QAT")
    p.add_argument("--no-freeze-embed", dest="freeze_embed", action="store_false")
    p.add_argument("--no-grad-checkpoint", dest="grad_checkpoint",
                   action="store_false", default=True)
    p.add_argument("--heldout-batches", type=int, default=32,
                   help="validation batches drawn from the DISJOINT val episode split")
    p.add_argument("--val-frac", type=float, default=0.05,
                   help="fraction of episodes reserved as a held-out validation split")
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--budget-min", type=int, default=210)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--resume", default=None)
    p.add_argument("--smoke", action="store_true", help="tiny run: 3 calib + 5 train steps")
    return p.parse_args()


def build_policy(args, device):
    import data as D
    from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config
    from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy
    from lerobot.configs.types import FeatureType
    try:
        from lerobot.datasets.utils import dataset_to_policy_features
    except Exception:  # noqa: BLE001
        from lerobot.policies.factory import dataset_to_policy_features

    _, ds_meta, _, _ = D.build_dataset_and_preprocessor(args)
    cfg = MolmoAct2Config(
        checkpoint_path=args.teacher, chunk_size=10, n_action_steps=10,
        action_mode="continuous", model_dtype="bfloat16", device=args.device,
    )
    feats = dataset_to_policy_features(ds_meta.features)
    cfg.output_features = {k: v for k, v in feats.items() if v.type is FeatureType.ACTION}
    cfg.input_features = {k: v for k, v in feats.items() if v.type is not FeatureType.ACTION}
    # Backbone QAT: gradients MUST reach the backbone via the action expert's
    # per-layer KV cross-attention, so knowledge-insulation is OFF (opposite of
    # the AE recipe, which froze the backbone).
    try:
        cfg.enable_knowledge_insulation = False
    except Exception:  # noqa: BLE001
        pass
    policy = MolmoAct2Policy(cfg).to(device)
    return policy


@torch.no_grad()
def eval_flow(policy, batches):
    policy.eval()
    tot = 0.0
    for b in batches:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss, _ = policy.forward(b)
        tot += float(loss)
    return tot / max(len(batches), 1)


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)
    logf = open(os.path.join(args.out, "train_log.jsonl"), "a")

    def log(d):
        d["t"] = time.time()
        logf.write(json.dumps(d) + "\n"); logf.flush()
        print("[llm-qat] " + " ".join(f"{k}={v}" for k, v in d.items() if k != "t"), flush=True)

    import data as D
    import llm_quant as LQ

    io_bits = None if args.io_bits is not None and args.io_bits < 0 else args.io_bits
    tag = f"W{args.weight_bits}A{args.act_bits}"
    groups = None if not args.groups else [g.strip() for g in args.groups.split(",") if g.strip()]
    log({"event": "start", "tag": tag, "io_bits": io_bits, "groups": groups,
         "target_steps": args.target_steps})

    policy = build_policy(args, device)
    backbone = policy._backbone()
    transformer = backbone.transformer

    # Proper DISJOINT validation split: train on the training episodes, validate on
    # held-out episodes never seen in training (the old 8-batch heldout was drawn
    # from the training stream -> leaky/noisy -> best.pt selected a memorized point).
    train_eps, val_eps = D.episode_split(args, val_frac=args.val_frac, seed=args.seed)
    full_iter = D.iter_full_batches(args, device, episodes_override=train_eps, shuffle=True)
    val_iter = D.iter_full_batches(args, device, episodes_override=val_eps, shuffle=False)
    heldout = [next(val_iter) for _ in range(args.heldout_batches)]
    log({"event": "split", "train_eps": len(train_eps), "val_eps": len(val_eps),
         "heldout_batches": args.heldout_batches})

    # FP reference (pre-quant)
    fp_base = eval_flow(policy, heldout)
    log({"event": "fp_baseline", "flow_loss": round(fp_base, 4)})

    # --- quantize the backbone in place ------------------------------------- #
    qdt = torch.float32 if args.quant_dtype == "fp32" else torch.bfloat16
    transformer.to(dtype=qdt)
    LQ.quantize_backbone_(transformer, weight_bits=args.weight_bits,
                          act_bits=args.act_bits, io_bits=io_bits, groups=groups)
    transformer.to(device=device, dtype=qdt)
    log({"event": "quantized", "quant_dtype": args.quant_dtype,
         **LQ.count_quant_linears(transformer)})

    # freeze everything except the backbone transformer (match by param identity)
    bb_ids = {id(p) for p in transformer.parameters()}
    for p in policy.parameters():
        p.requires_grad = id(p) in bb_ids
    if args.freeze_embed and hasattr(transformer, "wte"):
        for p in transformer.wte.parameters():
            p.requires_grad = False
    n_train = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    log({"event": "freeze", "trainable_backbone_params_M": round(n_train / 1e6, 2),
         "freeze_embed": bool(args.freeze_embed)})

    if args.grad_checkpoint and args.target_steps > 0:
        try:
            backbone.gradient_checkpointing_enable()
        except Exception:  # noqa: BLE001
            transformer.gradient_checkpointing = True
        log({"event": "grad_checkpoint", "enabled": True})

    opt = torch.optim.AdamW(
        [p for p in transformer.parameters() if p.requires_grad],
        lr=args.ae_lr, betas=(0.9, 0.95), weight_decay=args.wd,
    )

    best = float("inf")   # best held-out flow loss seen (checkpointed to best.pt)

    def save(step, name):
        path = os.path.join(args.out, name)
        blob = {
            "state_dict": {k: v.detach().cpu() for k, v in transformer.state_dict().items()},
            "weight_bits": args.weight_bits, "act_bits": args.act_bits,
            "io_bits": io_bits, "groups": groups, "step": step,
            "best": best, "kind": "backbone_qat",
        }
        if name == "latest.pt":
            blob["optimizer"] = opt.state_dict()
        torch.save(blob, path)
        return path

    start_step = 0
    resumed = bool(args.resume and os.path.exists(args.resume))
    if resumed:
        blob = torch.load(args.resume, map_location="cpu", weights_only=False)
        transformer.load_state_dict(blob["state_dict"], strict=True)
        # SEAMLESS RESUME: move the loaded (already-trained) params/buffers AND any
        # non-registered Brevitas scale caches onto the compute device, but do NOT
        # re-calibrate. Re-running activation calibration at every chunk boundary
        # perturbs the trained scales and compounds into divergence -- the resume
        # must be numerically identical to continuing the same process.
        transformer.to(device=device, dtype=qdt)
        dev = torch.device(device)
        moved = 0
        for m in transformer.modules():
            for nm, val in list(m.__dict__.items()):
                if torch.is_tensor(val) and val.device != dev:
                    setattr(m, nm, val.to(dev))
                    moved += 1
        if "optimizer" in blob:
            opt.load_state_dict(blob["optimizer"])
            for st in opt.state.values():         # AdamW moments load on CPU -> GPU
                for k, v in st.items():
                    if torch.is_tensor(v):
                        st[k] = v.to(dev)
        start_step = int(blob.get("step", 0))
        best = float(blob.get("best", best))
        # sanity: resumed flow must match the checkpoint (no recalibration drift)
        ho_res = eval_flow(policy, heldout)
        log({"event": "resume", "from": args.resume, "step": start_step,
             "best": round(best, 4), "resume_flow": round(ho_res, 4),
             "moved_tensors": moved})
    else:
        # PTQ calibration (fresh start only)
        from brevitas.graph.calibrate import calibration_mode
        n_cal = 3 if args.smoke else args.calib_batches
        policy.eval()
        with torch.no_grad(), calibration_mode(transformer):
            for _ in range(n_cal):
                b = next(full_iter)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    policy.forward(b)
        ptq = eval_flow(policy, heldout)
        best = ptq
        save(start_step, "best.pt")   # PTQ init is the max-so-far; always keep best.pt
        log({"event": "ptq_done", "calib_batches": n_cal, "flow_loss": round(ptq, 4),
             "best": round(best, 4), "fp": round(fp_base, 4)})

    # PTQ-only cell: calibrate, snapshot, done.
    if args.target_steps <= 0 and not args.smoke:
        final = save(0, f"llm_{tag}_ptq.pt")
        log({"event": "done_ptq", "ckpt": final})
        return

    target = 5 + start_step if args.smoke else args.target_steps

    def lr_scale(step):
        if step < args.warmup:
            return step / max(args.warmup, 1)
        if args.lr_schedule == "constant":
            return 1.0
        p = (step - args.warmup) / max(target - args.warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

    policy.train()
    t0 = time.time()
    running = 0.0
    step = start_step
    skipped = 0
    train_params = [p for p in transformer.parameters() if p.requires_grad]
    while step < target:
        step += 1
        for g in opt.param_groups:
            g["lr"] = args.ae_lr * lr_scale(step)
        opt.zero_grad(set_to_none=True)
        acc = 0.0
        for _ in range(args.grad_accum):
            b = next(full_iter)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss, _ = policy.forward(b)
            (loss / args.grad_accum).backward()
            acc += float(loss) / args.grad_accum
        gn = torch.nn.utils.clip_grad_norm_(train_params, args.clip)
        # Low-bit activation quant (esp. A<=4) can produce non-finite grads that
        # would corrupt params through the optimizer; skip those steps so training
        # stays stable instead of collapsing to NaN.
        if torch.isfinite(gn) and math.isfinite(acc):
            opt.step()
            running += acc
        else:
            skipped += 1
        opt.zero_grad(set_to_none=True)

        if step % args.log_every == 0:
            sps = (step - start_step) / (time.time() - t0)
            log({"step": step, "loss": round(running / args.log_every, 4),
                 "grad_norm": (round(float(gn), 3) if torch.isfinite(gn) else "nonfinite"),
                 "lr": round(opt.param_groups[0]["lr"], 7),
                 "skipped": skipped, "sps": round(sps, 3)})
            running = 0.0

        if step % args.eval_every == 0:
            ho = eval_flow(policy, heldout)
            improved = ho < best
            if improved:
                best = ho
                save(step, "best.pt")
            log({"step": step, "event": "heldout", "flow_loss": round(ho, 4),
                 "best": round(best, 4), "fp": round(fp_base, 4),
                 "saved_best": improved})
            policy.train()

        if step % args.save_every == 0 or step >= target:
            save(step, "latest.pt")

        if (time.time() - t0) / 60.0 >= args.budget_min and step < target:
            save(step, "latest.pt")
            log({"event": "budget_exit", "step": step, "target": target})
            return

    ho = eval_flow(policy, heldout)
    final = save(step, f"llm_{tag}_qat_ep.pt")
    save(step, "latest.pt")
    log({"event": "done", "step": step, "final_flow": round(ho, 4),
         "fp": round(fp_base, 4), "ckpt": final})


if __name__ == "__main__":
    main()
