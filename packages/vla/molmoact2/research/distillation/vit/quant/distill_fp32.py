"""Standalone fp32 seam-distillation driver (torchdistill-free).

The original distillation runs through torchdistill (``train.py`` + a YAML) inside
the ``molmoact2-vitdistill`` docker image. That image/cluster is not always
available; this driver reproduces the *same* recipe (frozen SigLIP2 teacher,
cosine + normalized-MSE seam loss, AdamW lr=1e-3 wd=0.05, cosine schedule, 10
epochs, batch 32) using only ``distill.teacher`` / ``distill.data`` /
``distill.losses`` + ``quant.variants`` -- all of which live in the eval SIF.

Used by the 2-bit size-vs-precision sweep to distill the grown students
(tinyvit_m 320x6, tinyvit_l 416x8); size S reuses the existing
``siglip_nano_full.pt``. Saves a torchdistill-compatible blob
(``{"model": SeamStudent.state_dict()}``) so ``quant.variants.load_encoder``
loads it unchanged.

torchrun (DDP, one process per GPU):
    torchrun --standalone --nproc_per_node=4 -m quant.distill_fp32 \
        --variant tinyvit_m --epochs 10 --batch 32 \
        --out /outputs/siglip_nano_m_full.pt
"""
from __future__ import annotations

import argparse
import os
import pathlib
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from distill.config import StudentConfig
from distill.losses import seam_cosine_loss, seam_norm_mse_loss
from distill.student import SeamStudent
from distill.teacher import build_seam_teacher
from quant.variants import VARIANTS, resolve


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def rank0() -> bool:
    return (not is_dist()) or dist.get_rank() == 0


def main() -> None:
    ap = argparse.ArgumentParser(description="fp32 seam distillation (torchdistill-free)")
    ap.add_argument("--variant", required=True, help="tinyvit_s|tinyvit_m|tinyvit_l|...")
    ap.add_argument("--checkpoint-path", default="allenai/MolmoAct2-LIBERO")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--target-steps", type=int, default=None,
                    help="train until this many optimizer steps (overrides epochs)")
    ap.add_argument("--prior-steps", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--subsample", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--max-minutes", type=float, default=None,
                    help="wall-clock budget: save a (non-final) checkpoint and exit "
                         "cleanly before this many minutes so a chunked SLURM job can "
                         "resubmit and resume (used on 2h-capped partitions)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if "RANK" in os.environ and torch.cuda.is_available():
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    variant = resolve(args.variant)
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; known={list(VARIANTS)}")
    cfg = StudentConfig(**VARIANTS[variant])

    teacher = build_seam_teacher(checkpoint_path=args.checkpoint_path,
                                 model_dtype="bfloat16", already_normalized=True).to(device).eval()
    seam = SeamStudent(cfg).to(device, torch.float32).train()

    # Resume-on-preempt: mi210 preempts/requeues jobs; reload the last save (all
    # ranks, BEFORE the DDP wrap) and continue from steps_done so requeues don't
    # restart from scratch. steps_done >= target_steps means truly finished.
    if os.path.exists(args.out):
        try:
            prev = torch.load(args.out, map_location="cpu", weights_only=False)
            done = int(prev.get("steps_done", 0))
            # final=True means the schedule completed (epoch- or step-based). A
            # budget stop saves final=False, so chunked runs still resume here.
            if prev.get("final"):
                if rank0():
                    print(f"[distill] {args.out} already final ({done} steps); nothing to do.", flush=True)
                return
            seam.load_state_dict(prev["model"])
            args.prior_steps = max(args.prior_steps, done)
            if rank0():
                print(f"[distill] RESUME <- {args.out} prior_steps={args.prior_steps}", flush=True)
        except Exception as e:  # noqa: BLE001
            if rank0():
                print(f"[distill] resume skipped ({e!r})", flush=True)

    if rank0():
        print(f"[distill] variant={variant} cfg dim={cfg.dim} attn={cfg.num_attn_blocks} "
              f"params={seam.num_params()/1e6:.2f}M device={device}", flush=True)
    if is_dist():
        student = DDP(seam, device_ids=[local_rank], output_device=local_rank)
    else:
        student = seam
    core = student.module if is_dist() else student

    from distill.data import DataConfig, LiberoFrameDataset
    ds = LiberoFrameDataset(
        DataConfig(checkpoint_path=args.checkpoint_path, patchify_mode="processor",
                   subsample_stride=args.subsample, max_frames=args.max_frames),
        train=True,
    )
    sampler = DistributedSampler(ds, shuffle=True) if is_dist() else None
    loader = DataLoader(ds, batch_size=args.batch, shuffle=(sampler is None), sampler=sampler,
                        num_workers=args.workers, pin_memory=True, drop_last=True)

    steps_per_epoch = max(1, len(loader))
    total_target = args.target_steps if args.target_steps else args.epochs * steps_per_epoch
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, total_target))
    for _ in range(min(args.prior_steps, total_target)):   # fast-forward LR on resume
        sched.step()

    out = pathlib.Path(args.out)
    if rank0():
        out.parent.mkdir(parents=True, exist_ok=True)

    def save(steps: int, final: bool = False) -> None:
        if not rank0():
            return
        tmp = out.with_suffix(out.suffix + ".tmp")
        torch.save({"model": core.state_dict(), "variant": variant,
                    "steps_done": steps, "target_steps": total_target, "final": final,
                    "recipe": "fp32_seam_distill", "lr": args.lr, "epochs": args.epochs}, tmp)
        os.replace(tmp, out)

    if rank0():
        print(f"[distill] steps/epoch={steps_per_epoch} target={total_target} lr={args.lr} "
              f"batch={args.batch} world={dist.get_world_size() if is_dist() else 1}", flush=True)

    step, done_total, ep, stop = 0, args.prior_steps, 0, False
    budget_hit = False
    t_start = time.time()
    budget_s = args.max_minutes * 60.0 if args.max_minutes else None
    while not stop:
        if sampler is not None:
            sampler.set_epoch(ep)
        for batch in loader:
            patches4d = batch[0].to(device)  # [B, crops, 729, 588] normalized
            with torch.no_grad():
                t = teacher(patches4d.to(torch.bfloat16))                 # [B,crops,N,D]
                bsz, crops, npat, ddim = t.shape
                t = t.reshape(-1, npat, ddim).float()
            s = student(patches4d).reshape(-1, npat, ddim).float()
            loss = seam_cosine_loss(s, t) + seam_norm_mse_loss(s, t)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if done_total < total_target:
                sched.step()
            step += 1
            done_total = args.prior_steps + step
            if rank0() and step % 100 == 0:
                with torch.no_grad():
                    cos = F.cosine_similarity(s.detach(), t, dim=-1).mean().item()
                print(f"[distill] ep{ep} step{step} total{done_total}/{total_target} "
                      f"loss={loss.item():.4f} cos={cos:.4f} lr={sched.get_last_lr()[0]:.2e}", flush=True)
            if args.save_every and step % args.save_every == 0:
                save(done_total)
            if done_total >= total_target:
                stop = True
                break
            # Wall-clock budget for 2h-capped partitions: agree across ranks (MAX
            # all-reduce) so DDP never desyncs, then save a non-final checkpoint and
            # exit cleanly for the chunk-chaining sbatch to resume.
            if budget_s is not None and step % 100 == 0:
                hit = 1.0 if (time.time() - t_start) >= budget_s else 0.0
                if is_dist():
                    flag = torch.tensor([hit], device=device)
                    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
                    hit = flag.item()
                if hit > 0:
                    budget_hit = True
                    stop = True
                    break
        ep += 1
    save(done_total, final=(not budget_hit))
    if rank0():
        if budget_hit:
            print(f"[distill] BUDGET_STOP -> {out} ({done_total}/{total_target} steps); resume next chunk", flush=True)
        else:
            print(f"[distill] DONE -> {out} ({done_total} steps)", flush=True)
    if is_dist():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
