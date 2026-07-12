"""Stage-1 curriculum distillation: LLM-only (Run D).

Freeze the MolmoAct2-LIBERO teacher; on real LIBERO inputs (real ViT features + text +
discrete state tokens fused by ``build_input_embeddings``) capture per-layer hidden,
per-layer raw KV and the final hidden, and train the thin-twin student (student.py) to
regress them. Teacher targets are computed ON THE FLY (no backward, bf16) each step --
dumping 37 hidden + 36 KV tensors per sample would be ~100s of MB/sample.

Runs inside the eval SIF (lerobot + molmoact2 + transformers).

Modes:
  --synthetic : no teacher/data; validates student + heads + losses + optimizer numerically
                (loss must fall on a fixed random target). Use for smoke.
  (default)   : real teacher + LIBERO batches.

Key integration points flagged with [CLUSTER] are validated on the SIF in llmd-stage1.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch

from student import build_student
import losses as L


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--w-hidden", type=float, default=1.0)
    ap.add_argument("--w-kv", type=float, default=2.0)      # KV is the action-expert conditioning path -> weight higher
    ap.add_argument("--w-final", type=float, default=1.0)
    ap.add_argument("--student-cfg", default="{}", help="JSON overrides for LLMStudentConfig")
    ap.add_argument("--teacher", default=os.environ.get("MM2_CKPT", "allenai/MolmoAct2-LIBERO"))
    ap.add_argument("--data-root", default=os.environ.get("LIBERO_ROOT", ""))
    ap.add_argument("--repo-id", default=os.environ.get("LIBERO_REPO", "allenai/MolmoAct2-LIBERO-Dataset"))
    ap.add_argument("--revision", default=os.environ.get("LIBERO_REV", "main"))
    ap.add_argument("--out", default="/outputs/llm_distill/stage1")
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=10)
    return ap.parse_args()


def _dtype(name):
    return torch.bfloat16 if name == "bf16" else torch.float32


# --------------------------------------------------------------------------- synthetic
class SyntheticSource:
    """Fixed random teacher targets -> the student must be able to overfit them
    (validates the module + loss + optimizer wiring without the teacher/data)."""

    def __init__(self, cfg, args, device, dt):
        self.cfg, self.args, self.device, self.dt = cfg, args, device, dt
        B, N = args.batch, args.seq
        self.embeds = torch.randn(B, N, cfg.teacher_hidden, device=device, dtype=dt)
        self.hidden = [torch.randn(B, N, cfg.teacher_hidden, device=device, dtype=dt)
                       for _ in range(cfg.num_layers + 1)]
        self.kv = [(torch.randn(B, cfg.num_kv_heads, N, 128, device=device, dtype=dt),
                    torch.randn(B, cfg.num_kv_heads, N, 128, device=device, dtype=dt))
                   for _ in range(cfg.num_layers)]
        self.last = torch.randn(B, N, cfg.teacher_hidden, device=device, dtype=dt)
        self.mask = torch.ones(B, N, device=device)

    def __call__(self):
        return {
            "inputs_embeds": self.embeds,
            "mask": self.mask,
            "positions": None,
            "attn_bias": None,
            "teacher": {"hidden": self.hidden, "kv": self.kv, "last": self.last},
        }


# --------------------------------------------------------------------------- real teacher
class TeacherSource:
    """[CLUSTER] Real teacher feature capture over LIBERO batches.

    Loads MolmoAct2 (frozen), builds fused inputs_embeds via ``build_input_embeddings``
    from real (images, text, discrete-state) batches, and runs the text transformer with
    ``output_hidden_states=True`` + per-layer KV collection. Validated on the SIF.
    """

    def __init__(self, cfg, args, device, dt):
        self.cfg, self.args, self.device, self.dt = cfg, args, device, dt
        from lerobot.policies.molmoact2.molmoact2_hf_model import (
            modeling_molmoact2 as HFM,
        )
        from llm_distill_data import iter_libero_batches, capture_teacher
        self._capture = capture_teacher
        print(f"[stage1] loading teacher {args.teacher}", flush=True)
        self.model = HFM.MolmoAct2ForConditionalGeneration.from_pretrained(
            args.teacher, torch_dtype=dt
        ).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.batches = iter_libero_batches(self.args, self.device)

    @torch.no_grad()
    def __call__(self):
        return self._capture(self.model, next(self.batches))


def main():
    args = get_args()
    device = torch.device(args.device)
    dt = _dtype(args.dtype)
    os.makedirs(args.out, exist_ok=True)

    student, heads, cfg = build_student(json.loads(args.student_cfg))
    student = student.to(device=device, dtype=dt)
    heads = heads.to(device=device, dtype=dt)
    student.train(); heads.train()

    n_s = sum(p.numel() for p in student.parameters())
    n_h = sum(p.numel() for p in heads.parameters())
    print(f"[stage1] student {n_s/1e6:.1f}M + distill heads {n_h/1e6:.1f}M "
          f"(heads dropped at deploy) | dtype={dt} device={device}", flush=True)

    src = SyntheticSource(cfg, args, device, dt) if args.synthetic \
        else TeacherSource(cfg, args, device, dt)

    params = list(student.parameters()) + list(heads.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, args.warmup)))
    w = {"hidden": args.w_hidden, "kv": args.w_kv, "final": args.w_final}

    t0 = time.time()
    first = None
    for step in range(args.steps):
        cap = src()
        out = student(cap["inputs_embeds"], attention_bias=cap["attn_bias"],
                      positions=cap["positions"])
        loss, metrics = L.stage1_loss(out, heads, cap["teacher"], cap["mask"], w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()
        if first is None:
            first = metrics["loss"]
        if step % args.log_every == 0 or step == args.steps - 1:
            dt_s = time.time() - t0
            print(f"[stage1] step {step:5d} loss {metrics['loss']:.4f} "
                  f"h {metrics['hidden']:.4f} kv {metrics['kv']:.4f} f {metrics['final']:.4f} "
                  f"| h_cos {metrics['hidden_cos']:.3f} f_cos {metrics['final_cos']:.3f} "
                  f"| {(step+1)/dt_s:.2f} it/s", flush=True)
        if args.save_every and (step + 1) % args.save_every == 0:
            torch.save({"student": student.state_dict(), "heads": heads.state_dict(),
                        "cfg": cfg.__dict__, "step": step + 1},
                       os.path.join(args.out, f"student_step{step+1}.pt"))

    torch.save({"student": student.state_dict(), "heads": heads.state_dict(),
                "cfg": cfg.__dict__, "step": args.steps},
               os.path.join(args.out, "student_final.pt"))
    if args.synthetic:
        drop = (first - metrics["loss"]) / max(first, 1e-9)
        ok = metrics["loss"] < first
        print(f"[stage1][synthetic] loss {first:.4f} -> {metrics['loss']:.4f} "
              f"({drop*100:.1f}% drop) : {'PASS' if ok else 'FAIL'}", flush=True)
    print(f"[stage1] done -> {args.out}/student_final.pt", flush=True)


if __name__ == "__main__":
    main()
