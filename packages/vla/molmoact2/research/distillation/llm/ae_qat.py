"""PTQ + QAT fine-tuning of the fake-quantized MolmoAct2 action expert.

Quantizes ONLY the action expert (ViT + LLM frozen), PTQ-calibrates the quant
scales on real LIBERO flow inputs, then QAT fine-tunes on the stock
flow-matching MSE (``policy.forward(batch)``) to recover accuracy at the target
W/A precision. No student / no distillation -- the objective is the original
task loss against demo actions.

Chunked + resumable: runs toward ``--target-steps`` in wall-clock ``--budget-min``
slices, checkpointing ``latest.pt`` (ae weights + optimizer + step) so a
dependency-chained Slurm job can resume.

    python ae_qat.py --weight-bits 8 --act-bits 8 --io-bits 8 \
        --target-steps 3000 --out /outputs/ae_quant/w8a8

Blob format (for eval rebuild):
    {"state_dict": ae.state_dict(), "weight_bits", "act_bits", "io_bits",
     "step", "kind": "action_expert_qat"}
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
                   help="pin I/O layers to this bit-width; -1 = quantize I/O to target too")
    p.add_argument("--groups", default=None,
                   help="comma list of quant groups to ablate "
                        "(self_attn,cross_attn,mlp,modulation,io); default=all")
    p.add_argument("--target-steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--ae-lr", type=float, default=5e-5)
    p.add_argument("--lr-schedule", default="constant", choices=["constant", "cosine"])
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--calib-batches", type=int, default=16)
    p.add_argument("--quant-dtype", default="bf16", choices=["bf16", "fp32"],
                   help="dtype for the fake-quant action expert (bf16 matches the "
                        "policy and avoids autocast dtype mixing; fp32 = max precision)")
    p.add_argument("--heldout-batches", type=int, default=8)
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
    # AE-only training: detach LLM KV so we don't retain the frozen backbone graph.
    try:
        cfg.enable_knowledge_insulation = True
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
        print("[ae-qat] " + " ".join(f"{k}={v}" for k, v in d.items() if k != "t"), flush=True)

    import data as D
    import ae_quant as AQ

    io_bits = None if args.io_bits is not None and args.io_bits < 0 else args.io_bits
    tag = f"W{args.weight_bits}A{args.act_bits}"
    log({"event": "start", "tag": tag, "io_bits": io_bits, "target_steps": args.target_steps})

    policy = build_policy(args, device)
    backbone = policy._backbone()
    ae = backbone._require_action_expert()

    full_iter = D.iter_full_batches(args, device)

    # pinned held-out batches for a stable flow-loss curve
    heldout = [next(full_iter) for _ in range(args.heldout_batches)]

    # FP reference (pre-quant) for context
    fp_base = eval_flow(policy, heldout)
    log({"event": "fp_baseline", "flow_loss": round(fp_base, 4)})

    # --- quantize the action expert in place -------------------------------- #
    qdt = torch.float32 if args.quant_dtype == "fp32" else torch.bfloat16
    ae.to(dtype=qdt)
    groups = None if not args.groups else [g.strip() for g in args.groups.split(",") if g.strip()]
    AQ.quantize_action_expert_(ae, weight_bits=args.weight_bits,
                               act_bits=args.act_bits, io_bits=io_bits, groups=groups)
    # layerwise_quantize + QSDPA create fresh quant modules on CPU (default fp32);
    # unify dtype/device so quant buffers match the activations they see.
    ae.to(device=device, dtype=qdt)
    log({"event": "quantized", "quant_dtype": args.quant_dtype, **AQ.count_quant_linears(ae)})

    # freeze everything except the AE (robust: match by param identity)
    ae_ids = {id(p) for p in ae.parameters()}
    for p in policy.parameters():
        p.requires_grad = id(p) in ae_ids
    n_train = sum(p.numel() for p in ae.parameters() if p.requires_grad)
    log({"event": "freeze", "trainable_ae_params_M": round(n_train / 1e6, 2)})

    opt = torch.optim.AdamW(
        [p for p in ae.parameters() if p.requires_grad],
        lr=args.ae_lr, betas=(0.9, 0.95), weight_decay=args.wd,
    )

    start_step = 0
    if args.resume and os.path.exists(args.resume):
        blob = torch.load(args.resume, map_location="cpu", weights_only=False)
        ae.load_state_dict(blob["state_dict"], strict=True)
        if "optimizer" in blob:
            opt.load_state_dict(blob["optimizer"])
        start_step = int(blob.get("step", 0))
        log({"event": "resume", "from": args.resume, "step": start_step})
    else:
        # PTQ calibration (fresh start only)
        from brevitas.graph.calibrate import calibration_mode
        n_cal = 3 if args.smoke else args.calib_batches
        policy.eval()
        with torch.no_grad(), calibration_mode(ae):
            for _ in range(n_cal):
                b = next(full_iter)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    policy.forward(b)
        ptq = eval_flow(policy, heldout)
        log({"event": "ptq_done", "calib_batches": n_cal, "flow_loss": round(ptq, 4),
             "fp": round(fp_base, 4)})

    target = 5 + start_step if args.smoke else args.target_steps

    def lr_scale(step):
        if step < args.warmup:
            return step / max(args.warmup, 1)
        if args.lr_schedule == "constant":
            return 1.0
        p = (step - args.warmup) / max(target - args.warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

    def save(step, name):
        path = os.path.join(args.out, name)
        blob = {
            "state_dict": {k: v.detach().cpu() for k, v in ae.state_dict().items()},
            "weight_bits": args.weight_bits, "act_bits": args.act_bits,
            "io_bits": io_bits, "groups": groups, "step": step,
            "kind": "action_expert_qat",
        }
        if name == "latest.pt":
            blob["optimizer"] = opt.state_dict()
        torch.save(blob, path)
        return path

    policy.train()
    backbone.transformer.eval() if hasattr(backbone, "transformer") else None
    t0 = time.time()
    running = 0.0
    best = float("inf")
    step = start_step
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
        gn = torch.nn.utils.clip_grad_norm_(
            [p for p in ae.parameters() if p.requires_grad], args.clip)
        opt.step()
        running += acc

        if step % args.log_every == 0:
            sps = (step - start_step) / (time.time() - t0)
            log({"step": step, "loss": round(running / args.log_every, 4),
                 "grad_norm": round(float(gn), 3), "lr": round(opt.param_groups[0]["lr"], 7),
                 "sps": round(sps, 3)})
            running = 0.0

        if step % args.eval_every == 0:
            ho = eval_flow(policy, heldout)
            best = min(best, ho)
            log({"step": step, "event": "heldout", "flow_loss": round(ho, 4),
                 "best": round(best, 4), "fp": round(fp_base, 4)})
            policy.train()

        if step % args.save_every == 0 or step >= target:
            save(step, "latest.pt")

        if (time.time() - t0) / 60.0 >= args.budget_min and step < target:
            save(step, "latest.pt")
            log({"event": "budget_exit", "step": step, "target": target})
            return

    ho = eval_flow(policy, heldout)
    final = save(step, f"ae_{tag}_qat_ep.pt")
    save(step, "latest.pt")
    log({"event": "done", "step": step, "final_flow": round(ho, 4),
         "fp": round(fp_base, 4), "ckpt": final})


if __name__ == "__main__":
    main()
