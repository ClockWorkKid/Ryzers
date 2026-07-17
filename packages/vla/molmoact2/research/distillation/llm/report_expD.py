"""Experiment D reporting: aggregate closed-loop success across suites + plot training curves.

Two independent jobs (either can be run alone):
  * --gate-dir DIR   : read DIR/eval_<suite>/eval_result.json for the 4 LIBERO suites, compute
                       overall success (mean over all episodes) + per-suite/per-task, write
                       DIR/summary.json and DIR/success_by_suite.png.
  * --train-log F:LABEL [F:LABEL ...] : parse train_joint train_log.jsonl files and plot the
                       flow (+anchor) loss curves to --out/loss_curves.png.

Everything stays tiny (rule 4). Matplotlib Agg, no GUI.
"""
from __future__ import annotations
import argparse, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]


def _pc(agg):
    if isinstance(agg, dict):
        if isinstance(agg.get("overall"), dict) and "pc_success" in agg["overall"]:
            return float(agg["overall"]["pc_success"])
        if "pc_success" in agg:
            return float(agg["pc_success"])
    return None


def aggregate_gate(gate_dir):
    rows, total_succ, total_ep = [], 0.0, 0
    for suite in SUITES:
        p = os.path.join(gate_dir, f"eval_{suite}", "eval_result.json")
        if not os.path.exists(p):
            rows.append({"suite": suite, "pc_success": None, "n_episodes": 0, "missing": True})
            continue
        with open(p) as f:
            r = json.load(f)
        agg = r.get("aggregated", r)
        pc = _pc(agg)
        n_tasks = len(r.get("task_ids", [])) or 1
        n_ep = int(r.get("n_episodes", 0)) * n_tasks
        rows.append({"suite": suite, "pc_success": pc, "n_episodes": n_ep})
        if pc is not None:
            total_succ += pc / 100.0 * n_ep
            total_ep += n_ep
    overall = round(100.0 * total_succ / total_ep, 2) if total_ep else None
    summary = {"gate_dir": gate_dir, "overall_pc_success": overall,
               "total_episodes": total_ep, "baseline_pc_success": 30.0, "suites": rows}
    with open(os.path.join(gate_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    labels = [r["suite"].replace("libero_", "") for r in rows]
    vals = [r["pc_success"] if r["pc_success"] is not None else 0 for r in rows]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    bars = ax.bar(labels, vals, color="#4C72B0")
    ax.axhline(30.0, ls="--", c="crimson", lw=1.2, label="LLM-only baseline 30%")
    if overall is not None:
        ax.axhline(overall, ls="-", c="green", lw=1.2, label=f"overall {overall}%")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}", ha="center", fontsize=8)
    ax.set_ylabel("pc_success (%)"); ax.set_ylim(0, 100)
    ax.set_title(f"Experiment D closed-loop success ({os.path.basename(gate_dir)})")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(gate_dir, "success_by_suite.png"), dpi=110)
    print(f"[report] gate={gate_dir} overall={overall}% n_ep={total_ep} -> summary.json + png")
    return summary


def _read_log(path):
    steps, flow, anchor = [], [], []
    with open(path) as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            if "step" in d and "flow_loss" in d and d.get("event") is None:
                steps.append(d["step"]); flow.append(d["flow_loss"])
                anchor.append(d.get("anchor_loss", 0.0))
    return steps, flow, anchor


def plot_train(logs, out):
    fig, ax = plt.subplots(figsize=(7, 4))
    for spec in logs:
        path, label = spec.split(":", 1) if ":" in spec else (spec, os.path.basename(spec))
        if not os.path.exists(path):
            print(f"[report] WARN missing train log {path}")
            continue
        s, fl, an = _read_log(path)
        if not s:
            continue
        ax.plot(s, fl, label=f"{label} flow")
        if any(a > 0 for a in an):
            ax.plot(s, an, ls="--", alpha=0.7, label=f"{label} anchor")
    ax.set_xlabel("step"); ax.set_ylabel("loss"); ax.set_title("Experiment D training curves")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, "loss_curves.png")
    fig.savefig(p, dpi=110)
    print(f"[report] train curves -> {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate-dir", default="")
    ap.add_argument("--train-log", nargs="*", default=[],
                    help="path or path:label of train_log.jsonl files to overlay")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if args.gate_dir:
        aggregate_gate(args.gate_dir)
    if args.train_log:
        plot_train(args.train_log, args.out or (args.gate_dir or "."))


if __name__ == "__main__":
    main()
