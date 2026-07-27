"""Quantization-aware distillation (QAT recovery) for the student.

When PTQ W4A6 loses seam fidelity, fine-tune the fake-quant student with STE
against the FROZEN teacher using the existing seam loss (cosine + normalized-MSE,
from distill.losses). This is "distill-during-quantization": the student learns
weights that are robust to the 4-bit/6-bit fake-quant. The data-mix lever
(LIBERO-only vs LIBERO+DROID) mirrors distill_full.yaml / distill_hybrid_droid.yaml.

Methodology note (mentor's MiniLLM reference): MiniLLM's reverse-KLD is specific
to LLM next-token distributions; our seam target is a feature-regression, so we
adopt the *methodology* (distill while compressing + richer real-data mix), not
the reverse-KLD loss.

    torchrun --nproc_per_node=4 -m quant.qat --variant cnn \
        --ckpt resource/ckpt/vit_distill/cnn_fpga_full.pt \
        --weight-bits 4 --act-bits 6 --data mix --epochs 3 \
        --out resource/ckpt/vit_distill/quant/cnn_w4a6_qat.pt
"""
from __future__ import annotations

import argparse
import copy
import os
import pathlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from brevitas.graph.calibrate import calibration_mode

from distill.losses import seam_cosine_loss, seam_norm_mse_loss, seam_downstream_loss
from distill.teacher import build_seam_teacher
from quant.quant_student import quantize_student_
from quant.variants import load_encoder, resolve

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_aug(aug_mode: str):
    """Map an --aug preset to a distill.data.AugConfig.

    heavy = the original recipe (strong color jitter/blur/noise/erase); mild =
    light photometric only (a cheap ablation, since heavy input-space aug may
    push the encoder off the rollout distribution the frozen policy expects);
    none = clean frames (patchify+normalize only).
    """
    from distill.data import AugConfig
    if aug_mode == "none":
        return AugConfig(enable=False)
    if aug_mode == "mild":
        return AugConfig(enable=True, color_jitter=0.1, hue=0.02, blur_p=0.1,
                         gray_p=0.0, noise_std=0.0, erase_p=0.0, hflip_p=0.0)
    return AugConfig()  # heavy (default recipe)


def build_loader(data: str, checkpoint_path: str, batch: int, workers: int,
                 subsample: int, max_frames: int | None, droid_root: str,
                 aug_mode: str = "heavy") -> DataLoader:
    from distill.data import DataConfig, LiberoFrameDataset
    aug = build_aug(aug_mode)
    if data == "mix":
        from distill.data import LiberoDroidMixDataset
        ds = LiberoDroidMixDataset(
            train=True, checkpoint_path=checkpoint_path, patchify_mode="processor",
            subsample_stride=subsample, max_frames=max_frames, droid_root=droid_root,
        )
    else:
        cfg = DataConfig(checkpoint_path=checkpoint_path, patchify_mode="processor",
                         subsample_stride=subsample, max_frames=max_frames, aug=aug)
        ds = LiberoFrameDataset(cfg, train=True)
    return DataLoader(ds, batch_size=batch, shuffle=True, num_workers=workers,
                      pin_memory=True, drop_last=True)


def build_pooled_idx(checkpoint_path: str) -> torch.Tensor:
    """Compute the frozen 2x2 pooling index (``image_token_pooling``) once.

    The pooled-patch layout depends only on the (fixed) LIBERO input geometry, so
    a single processor call on a dummy 256x256 frame yields the exact
    ``pooled_patches_idx`` the model's ``forward`` uses to gather 2x2 groups. Shape
    ``[T, pool_dim]`` with -1 marking padded (border) patches.
    """
    from PIL import Image
    from distill.teacher import load_image_processor
    proc = load_image_processor(checkpoint_path)
    img_proc = getattr(proc, "image_processor", proc)
    dummy = Image.fromarray(np.full((256, 256, 3), 127, dtype=np.uint8))
    out = img_proc.preprocess(images=[dummy], return_tensors="pt")
    idx = out["image_token_pooling"]
    if not torch.is_tensor(idx):
        idx = torch.as_tensor(np.asarray(idx))
    idx = idx.long()
    while idx.dim() > 2:  # drop any leading batch dim
        idx = idx[0]
    return idx  # [T, pool_dim]


class DownstreamHead(nn.Module):
    """Frozen 2x2 attention-pool + vision->LLM projector, applied to a seam.

    Wraps deep copies of the teacher backbone's ``image_pooling_2d`` and
    ``image_projector`` (fp32, sdpa attention so the border pooling mask is
    honored) plus a fixed ``pooled_patches_idx``. ``forward(seam)`` reproduces
    ``MolmoAct2VisionBackbone.forward``'s pool+project path (minus dropout) and
    returns the tokens the LLM consumes, so student/teacher can be matched there.
    """

    def __init__(self, backbone: nn.Module, pooled_idx: torch.Tensor, device) -> None:
        super().__init__()
        self.pool = copy.deepcopy(backbone.image_pooling_2d).to(device, torch.float32).eval()
        self.proj = copy.deepcopy(backbone.image_projector).to(device, torch.float32).eval()
        # sdpa honors the padded-patch attn_mask (the eager path silently ignores it);
        # float32 for stable grads through the fake-quant student.
        self.pool.attn_implementation = "sdpa"
        self.pool.float32_attention = True
        self.use_mask = bool(getattr(backbone.adapter_config, "pooling_attention_mask", False))
        for p in self.parameters():
            p.requires_grad_(False)
        self.register_buffer("pooled_idx", pooled_idx.long())

    def forward(self, seam: torch.Tensor, tok_perm: torch.Tensor | None = None):
        # seam: [B, crops, N, dim] -> flatten crops so pooled_idx indexes crops*N.
        b = seam.shape[0]
        dim = seam.shape[-1]
        idx = self.pooled_idx if tok_perm is None else self.pooled_idx[tok_perm]
        t, p = idx.shape
        flat = seam.reshape(b, -1, dim).float()
        idx_b = idx.unsqueeze(0).expand(b, t, p)
        valid = idx_b >= 0
        batch_idx = torch.arange(b, device=seam.device).view(b, 1, 1).expand(b, t, p)
        to_pool = flat[batch_idx, idx_b.clamp(min=0)]           # [B,T,P,dim]
        to_pool = to_pool * valid.to(flat.dtype).unsqueeze(-1)
        to_pool = to_pool.reshape(b * t, p, dim)
        if self.use_mask:
            attn_mask = valid.reshape(b * t, 1, 1, p)
            denom = valid.reshape(b * t, p).float().sum(-1)
            denom = torch.where(denom == 0, torch.ones_like(denom), denom)
            query = to_pool.sum(-2, keepdim=True) / denom[:, None, None]
        else:
            attn_mask = None
            query = to_pool.mean(-2, keepdim=True)
        pooled = self.pool(query, to_pool, attn_mask=attn_mask)  # [B*T,1,H]
        pooled = pooled.reshape(b, t, -1)
        tokens = self.proj(pooled)                              # [B,T,text_hidden]
        token_valid = valid.any(-1)                             # [B,T]
        return tokens, token_valid


def main() -> None:
    ap = argparse.ArgumentParser(description="QAT recovery via seam distillation")
    ap.add_argument("--variant", default="cnn")
    ap.add_argument("--ckpt", required=True, help="distilled fp32 student checkpoint")
    ap.add_argument("--checkpoint-path", default="allenai/MolmoAct2-LIBERO")
    ap.add_argument("--weight-bits", type=int, default=4)
    ap.add_argument("--act-bits", type=int, default=6)
    ap.add_argument("--data", default="libero", choices=["libero", "mix"])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lr-schedule", default="cosine", choices=["cosine", "constant"],
                    help="constant is chunk-friendly for multi-job full-epoch chains")
    ap.add_argument("--target-steps", type=int, default=None,
                    help="train until cumulative (prior-steps + this-job) reach this; "
                         "enables 4h-walltime resume chains for full-epoch training")
    ap.add_argument("--prior-steps", type=int, default=0, help="steps already done in earlier chunks")
    ap.add_argument("--save-every", type=int, default=1000, help="checkpoint every N steps (timeout-safe)")
    ap.add_argument("--cosine-w", type=float, default=1.0)
    ap.add_argument("--norm-mse-w", type=float, default=1.0)
    ap.add_argument("--downstream-w", type=float, default=0.0,
                    help="weight on the downstream-consistency loss (match student vs "
                         "teacher AFTER the frozen 2x2 pool + vision->LLM projector, i.e. "
                         "the tokens the policy actually consumes). 0 -> off (raw-seam only)")
    ap.add_argument("--downstream-max-tokens", type=int, default=256,
                    help="random subset of pooled tokens/frame used to estimate the "
                         "downstream loss (bounds memory; 0 -> use all tokens)")
    ap.add_argument("--aug", default="heavy", choices=["heavy", "mild", "none"],
                    help="input-space augmentation strength (heavy = original recipe)")
    ap.add_argument("--calib-batches", type=int, default=16, help="PTQ init before QAT")
    ap.add_argument("--subsample", type=int, default=1, help="keep every Nth frame")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None, help="cap optimizer steps")
    ap.add_argument("--droid-root", default="/cache/droid_frames")
    ap.add_argument("--resume-quant", default=None,
                    help="resume from a prior quant blob (state_dict of the fake-quant "
                         "student). Enables two-stage recipes: DROID re-distill -> QAT polish.")
    ap.add_argument("--stage", default="qat", help="tag written to the output blob (e.g. redistill_droid, qat_ft)")
    ap.add_argument("--skip-calib", action="store_true",
                    help="skip PTQ init (use when resuming from an already-calibrated quant blob)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    variant = resolve(args.variant)
    teacher = build_seam_teacher(checkpoint_path=args.checkpoint_path,
                                 model_dtype="bfloat16", already_normalized=True).to(DEVICE).eval()

    student = load_encoder(variant, args.ckpt).to(DEVICE, torch.float32).eval()
    student = quantize_student_(student, weight_bits=args.weight_bits, act_bits=args.act_bits).to(DEVICE)
    if args.resume_quant:
        blob = torch.load(args.resume_quant, map_location=DEVICE, weights_only=False)
        miss, unexp = student.load_state_dict(blob["state_dict"], strict=False)
        student = student.to(DEVICE)
        # continuation of the SAME stage -> carry cumulative step count for target logic
        if args.prior_steps == 0 and blob.get("stage") == args.stage:
            args.prior_steps = int(blob.get("steps_done", 0))
        print(f"[qat] resumed quant weights <- {args.resume_quant} (miss={len(miss)} unexp={len(unexp)}, "
              f"prior_stage={blob.get('stage', blob.get('mode','?'))}, prior_steps={args.prior_steps})")
        # fast no-op for surplus chain chunks: this stage already converged/finished.
        if blob.get("stage") == args.stage and blob.get("final", False):
            print(f"[qat] stage {args.stage} already final (steps_done={blob.get('steps_done')}); nothing to do.")
            return

    loader = build_loader(args.data, args.checkpoint_path, args.batch, args.workers,
                          args.subsample, args.max_frames, args.droid_root, aug_mode=args.aug)

    # Downstream-consistency head (frozen pool+projector from the teacher backbone).
    head = None
    if args.downstream_w > 0:
        pooled_idx = build_pooled_idx(args.checkpoint_path).to(DEVICE)
        head = DownstreamHead(teacher.backbone, pooled_idx, DEVICE)
        print(f"[qat] downstream head ON w={args.downstream_w} pooled_tokens={pooled_idx.shape[0]} "
              f"pool_dim={pooled_idx.shape[1]} max_tokens={args.downstream_max_tokens}", flush=True)

    def to3d(p: torch.Tensor) -> torch.Tensor:
        return p.reshape(-1, p.shape[-2], p.shape[-1]) if p.dim() == 4 else p

    # PTQ init: set activation scales from data before QAT (stabilizes the STE start).
    if args.skip_calib:
        print("[qat] skip PTQ init (resuming calibrated quant blob)")
    else:
        print(f"[qat] PTQ init on {args.calib_batches} batches ...")
        student.eval()
        with torch.no_grad(), calibration_mode(student):
            for i, batch in enumerate(loader):
                patches = to3d(batch[0]).to(DEVICE, torch.float32)
                student(patches[: args.batch])
                if i + 1 >= args.calib_batches:
                    break

    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.05)
    steps_per_epoch = max(1, len(loader))
    total_target = args.target_steps if args.target_steps else args.epochs * steps_per_epoch
    if args.lr_schedule == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, total_target))
    else:
        sched = None
    student.train()
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prog = out.with_suffix(out.suffix + ".progress")

    def save(total_steps: int, final: bool = False) -> None:
        student.eval()
        tmp = out.with_suffix(out.suffix + ".tmp")
        torch.save({
            "state_dict": student.state_dict(),
            "variant": variant, "weight_bits": args.weight_bits, "act_bits": args.act_bits,
            "data": args.data, "epochs": args.epochs, "src_ckpt": args.ckpt, "mode": "qat",
            "stage": args.stage, "resumed_from": args.resume_quant, "lr": args.lr,
            "downstream_w": args.downstream_w, "aug": args.aug, "lr_schedule": args.lr_schedule,
            "steps_done": total_steps, "target_steps": total_target, "final": final,
        }, tmp)
        os.replace(tmp, out)
        prog.write_text(str(total_steps))
        student.train()

    print(f"[qat] W{args.weight_bits}A{args.act_bits} {variant} data={args.data} aug={args.aug} "
          f"steps/epoch={steps_per_epoch} prior={args.prior_steps} target={total_target} "
          f"lr={args.lr}({args.lr_schedule}) cos_w={args.cosine_w} nmse_w={args.norm_mse_w} "
          f"ds_w={args.downstream_w}", flush=True)

    step = 0            # steps THIS job
    done_total = args.prior_steps
    stop = False
    ep = 0
    while not stop:
        for batch in loader:
            patches4d = batch[0].to(DEVICE)        # [B, crops, 729, 588] normalized
            with torch.no_grad():
                t4d = teacher(patches4d.to(torch.bfloat16))        # [B, crops, N, D]
                bsz, crops, npat, ddim = t4d.shape
                t = t4d.reshape(-1, npat, ddim).float()
            s = student(to3d(patches4d).to(torch.float32)).float()  # [B*crops, N, D]
            loss = args.cosine_w * seam_cosine_loss(s, t) + args.norm_mse_w * seam_norm_mse_loss(s, t)
            ds_val = 0.0
            if head is not None:
                tok_perm = None
                if args.downstream_max_tokens and args.downstream_max_tokens < head.pooled_idx.shape[0]:
                    tok_perm = torch.randperm(head.pooled_idx.shape[0], device=DEVICE)[: args.downstream_max_tokens]
                with torch.no_grad():
                    t_tok, tvalid = head(t4d.float(), tok_perm=tok_perm)
                s_tok, _ = head(s.reshape(bsz, crops, npat, ddim), tok_perm=tok_perm)
                ds_loss = seam_downstream_loss(s_tok, t_tok, valid=tvalid)
                loss = loss + args.downstream_w * ds_loss
                ds_val = float(ds_loss.detach())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if sched is not None and done_total < total_target:
                sched.step()
            step += 1
            done_total = args.prior_steps + step
            if step % 50 == 0:
                with torch.no_grad():
                    cos = F.cosine_similarity(s.detach(), t, dim=-1).mean().item()
                lr = sched.get_last_lr()[0] if sched is not None else args.lr
                dstr = f" ds={ds_val:.4f}" if head is not None else ""
                print(f"[qat] ep{ep} step{step} total{done_total}/{total_target} "
                      f"loss={loss.item():.4f} cos={cos:.4f}{dstr} lr={lr:.2e}", flush=True)
            if args.save_every and step % args.save_every == 0:
                save(done_total)
            if (args.max_steps and step >= args.max_steps) or done_total >= total_target:
                stop = True
                break
        ep += 1
    save(done_total, final=(done_total >= total_target))
    print(f"[qat] saved -> {out}")


if __name__ == "__main__":
    main()
