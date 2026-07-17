"""Minitron curriculum driver (experiment C): prune -> heal -> eval -> gate, in a loop.

Anti-catastrophe by construction: we take SMALL width steps, heal each with distillation
(flow + all-layer teacher-KV anchoring to the ORIGINAL teacher), evaluate, and only proceed
to a more aggressive target if the current one healed successfully. On failure we stop and
keep the last accepted checkpoint (rollback). Faithful to Minitron's iterative recipe
("compress ~30%, distill, compress again" -- NVIDIA Model-Optimizer >50% guidance).

Each stage is a fresh subprocess (robust; reuses the validated stage scripts):
  step 1 : minitron_prune.py  (prune-from-TEACHER, with the full-width==teacher gate)
  step k : minitron_prune.py --from-ckpt <prev heal latest.pt>  (prune-from-CURRENT healed)
  heal   : train_joint.py     (SHORT anchored distillation; builds student from ckpt cfg)

Gate signal is self-consistent PER heal process: train_joint logs the teacher baseline and the
student held-out flow on the SAME pinned held-out batches, so we gate on
  best_heldout_flow <= teacher_baseline * (1 + tol).
"""

from __future__ import annotations
import argparse, json, os, subprocess, sys, time


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedule", required=True,
                    help="comma list of 'interm:heads' targets, most->least aggressive is NOT "
                         "required; give them monotonically decreasing e.g. "
                         "'8192:32,7168:32,6144:32,5120:24,4096:24'")
    ap.add_argument("--importance", default="/outputs/llm_distill/minitron/importance.pt")
    ap.add_argument("--root", default="/outputs/llm_distill/minitron/curriculum")
    ap.add_argument("--heal-steps", type=int, default=600)
    ap.add_argument("--heal-lr", default="1e-4")
    ap.add_argument("--heal-lora-lr", default="2e-4")
    ap.add_argument("--anchor-weight", default="1.0")
    ap.add_argument("--anchor-mode", default="both")
    ap.add_argument("--anchor-beta", default="1.0")
    ap.add_argument("--lora-rank", default="16")
    ap.add_argument("--batch", default="4")
    ap.add_argument("--grad-accum", default="2")
    ap.add_argument("--student-dtype", default="bf16", choices=["fp32", "bf16"])
    ap.add_argument("--grad-checkpoint", action="store_true", default=False)
    ap.add_argument("--eval-batches", default="8")
    ap.add_argument("--tol", type=float, default=0.15,
                    help="accept if best healed heldout flow <= teacher_baseline*(1+tol)")
    ap.add_argument("--num-workers", default="8")
    return ap.parse_args()


def run(cmd, logpath):
    print(f"[curriculum] $ {' '.join(cmd)}", flush=True)
    with open(logpath, "w") as lf:
        p = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
    print(f"[curriculum]   -> rc={p.returncode} (log {logpath})", flush=True)
    return p.returncode


def read_heal_gate(train_log):
    """Return (teacher_baseline, best_heldout_flow) from a train_joint train_log.jsonl."""
    tb, best = None, None
    with open(train_log) as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("event") == "teacher_baseline":
                tb = float(d["flow_loss"])
            if d.get("event") == "heldout" and "heldout_flow" in d:
                hf = float(d["heldout_flow"])
                best = hf if best is None else min(best, hf)
    return tb, best


def main():
    args = get_args()
    os.makedirs(args.root, exist_ok=True)
    summary = {"schedule": args.schedule, "tol": args.tol, "stages": []}
    sched = []
    for tok in args.schedule.split(","):
        i, h = tok.split(":")
        sched.append((int(i), int(h)))

    here = os.path.dirname(os.path.abspath(__file__))
    prune_py = os.path.join(here, "minitron_prune.py")
    train_py = os.path.join(here, "train_joint.py")

    current_ckpt = None          # None -> prune from teacher; else prune from this healed student
    accepted = []                # list of accepted heal latest.pt
    for k, (interm, heads) in enumerate(sched, start=1):
        tag = f"step{k}_i{interm}_h{heads}"
        sdir = os.path.join(args.root, tag)
        os.makedirs(sdir, exist_ok=True)
        pruned = os.path.join(sdir, "pruned.pt")
        preport = os.path.join(sdir, "prune_report.json")
        heal_dir = os.path.join(sdir, "heal")

        # ---- prune ------------------------------------------------------------
        pcmd = [sys.executable, prune_py,
                "--target-intermediate", str(interm), "--target-heads", str(heads),
                "--out", pruned, "--report", preport,
                "--batch", args.batch, "--eval-batches", args.eval_batches]
        if current_ckpt is None:
            pcmd += ["--importance", args.importance]
        else:
            pcmd += ["--from-ckpt", current_ckpt]
        rc = run(pcmd, os.path.join(sdir, "prune.log"))
        stage = {"step": k, "interm": interm, "heads": heads,
                 "source": "teacher" if current_ckpt is None else current_ckpt}
        if rc != 0 or not os.path.exists(pruned):
            stage["status"] = "PRUNE_FAILED"
            summary["stages"].append(stage)
            print(f"[curriculum] STOP: prune failed at {tag}", flush=True)
            break
        try:
            pr = json.load(open(preport))
            stage["pruned_init_flow_gap"] = pr.get("flow_gap_vs_teacher")
            stage["pruned_init_kv_cos_v_mean"] = pr.get("pruned_init_kv_cos_v_mean")
            stage["approx_reduction_x"] = pr.get("approx_reduction_x")
        except Exception:
            pr = {}

        # ---- heal (short, anchored) -------------------------------------------
        hcmd = [sys.executable, train_py, "--preset", "minitron", "--init", pruned,
                "--out-dir", heal_dir, "--steps", str(args.heal_steps),
                "--batch", args.batch, "--grad-accum", args.grad_accum,
                "--lr", args.heal_lr, "--lora-lr", args.heal_lora_lr,
                "--eval-every", "100", "--save-every", str(max(100, args.heal_steps // 2)),
                "--log-every", "20", "--lora-rank", args.lora_rank,
                "--num-workers", args.num_workers, "--heldout-batches", args.eval_batches,
                "--student-dtype", args.student_dtype,
                "--anchor-weight", args.anchor_weight, "--anchor-mode", args.anchor_mode,
                "--anchor-beta", args.anchor_beta]
        if args.grad_checkpoint:
            hcmd.append("--grad-checkpoint")
        rc = run(hcmd, os.path.join(sdir, "heal.log"))
        train_log = os.path.join(heal_dir, "train_log.jsonl")
        latest = os.path.join(heal_dir, "latest.pt")
        if rc != 0 or not os.path.exists(latest):
            stage["status"] = "HEAL_FAILED"
            summary["stages"].append(stage)
            print(f"[curriculum] STOP: heal failed at {tag}", flush=True)
            break
        tb, best = read_heal_gate(train_log)
        stage["teacher_baseline"] = tb
        stage["best_heldout_flow"] = best
        thresh = (tb * (1 + args.tol)) if tb is not None else None
        stage["accept_threshold"] = thresh
        accept = (best is not None and thresh is not None and best <= thresh)
        stage["status"] = "ACCEPTED" if accept else "REJECTED"
        summary["stages"].append(stage)
        print(f"[curriculum] {tag}: teacher={tb} best_heldout={best} thresh={thresh} "
              f"=> {stage['status']}", flush=True)
        # persist summary each stage (resumable inspection)
        json.dump(summary, open(os.path.join(args.root, "curriculum_summary.json"), "w"), indent=2)
        if not accept:
            print(f"[curriculum] STOP (anti-catastrophe): {tag} did not heal within tol; "
                  f"keeping last accepted {current_ckpt}", flush=True)
            break
        current_ckpt = latest
        accepted.append(latest)

    summary["accepted_chain"] = accepted
    summary["final_ckpt"] = current_ckpt
    json.dump(summary, open(os.path.join(args.root, "curriculum_summary.json"), "w"), indent=2)
    print(f"[curriculum] DONE. accepted {len(accepted)} stage(s). final={current_ckpt}", flush=True)


if __name__ == "__main__":
    main()
