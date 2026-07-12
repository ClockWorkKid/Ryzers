"""Aggregate closed-loop LIBERO results for teacher vs student and render a
comparison table + grouped bar chart.

Reads eval_info.json per (suite,task) unit under two result dirs (teacher,
student), computes per-suite and overall pc_success from per-episode successes,
and writes:
  artifacts/vit_distill/closedloop/compare.csv
  artifacts/vit_distill/closedloop/compare.png

Per workspace rule 2.a: numeric comparison overlays baseline (teacher) and
prediction (student) on the same axes.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
SUITE_LABEL = {
    "libero_spatial": "Spatial",
    "libero_object": "Object",
    "libero_goal": "Goal",
    "libero_10": "Long",
}


def collect(root: Path, mode: str) -> dict:
    """Return {suite: [per-episode success bools...]} from eval_info.json units."""
    out = {s: [] for s in SUITES}
    for unit in sorted(root.glob(f"eval_{mode}_*")):
        info = unit / "eval_info.json"
        if not info.exists():
            continue
        data = json.loads(info.read_text())
        for pt in data.get("per_task", []):
            suite = pt["task_group"]
            if suite in out:
                out[suite].extend(bool(x) for x in pt["metrics"]["successes"])
    return out


def pc(xs: list) -> float:
    return 100.0 * float(np.mean(xs)) if xs else float("nan")


def main() -> None:
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("artifacts/vit_distill/closedloop")
    t_root = base / "vd_eval_teacher"
    s_root = base / "vd_eval_student"
    teacher = collect(t_root, "teacher")
    student = collect(s_root, "student")

    rows = []
    for s in SUITES:
        rows.append((SUITE_LABEL[s], pc(teacher[s]), len(teacher[s]), pc(student[s]), len(student[s])))
    all_t = [v for s in SUITES for v in teacher[s]]
    all_s = [v for s in SUITES for v in student[s]]
    rows.append(("Overall", pc(all_t), len(all_t), pc(all_s), len(all_s)))

    out_dir = Path("artifacts/vit_distill/closedloop")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv = out_dir / "compare.csv"
    with csv.open("w") as f:
        f.write("suite,teacher_pc_success,teacher_n,student_pc_success,student_n,delta\n")
        for label, tpc, tn, spc, sn in rows:
            f.write(f"{label},{tpc:.1f},{tn},{spc:.1f},{sn},{spc - tpc:+.1f}\n")
    print(csv.read_text())

    labels = [r[0] for r in rows]
    tvals = [r[1] for r in rows]
    svals = [r[3] for r in rows]
    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - w / 2, tvals, w, label="Teacher (MolmoAct2 ViT)", color="#4C72B0")
    b2 = ax.bar(x + w / 2, svals, w, label="Student (distilled, 2.8M)", color="#DD8452")
    ax.set_ylabel("Success rate (%)")
    ax.set_title("LIBERO closed-loop: teacher vs distilled-student ViT (100 ep, seed 1000, MI210)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 105)
    ax.legend()
    ax.axvline(len(labels) - 1.5, ls="--", c="gray", lw=0.8)
    for b in list(b1) + list(b2):
        h = b.get_height()
        if not np.isnan(h):
            ax.annotate(f"{h:.0f}", (b.get_x() + b.get_width() / 2, h),
                        ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    png = out_dir / "compare.png"
    fig.savefig(png, dpi=130)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
