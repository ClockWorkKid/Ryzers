"""3-way closed-loop LIBERO comparison: teacher (stock MolmoAct2 ViT) vs
distilled-only student vs finetuned distilled-full model.

Reads eval_info.json per (suite,task) unit under three result dirs and computes
per-suite + overall pc_success from per-episode successes. Stdlib-only so it can
run inside the eval SIF or on any cluster login node.

Usage:
  python aggregate_ft_compare.py [BASE_DIR]
    BASE_DIR contains vd_eval_teacher/, vd_eval_student/, vd_eval_ft/
    (default: /outputs)
Writes BASE_DIR/compare_ft.csv and prints a table.
"""

import json
import sys
from pathlib import Path

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
LABEL = {"libero_spatial": "Spatial", "libero_object": "Object",
         "libero_goal": "Goal", "libero_10": "Long"}
ARMS = [("teacher", "vd_eval_teacher"), ("student", "vd_eval_student"), ("ft", "vd_eval_ft")]


def collect(root: Path, mode: str) -> dict:
    out = {s: [] for s in SUITES}
    if not root.exists():
        return out
    for unit in sorted(root.glob(f"eval_{mode}_*")):
        info = unit / "eval_info.json"
        if not info.exists():
            continue
        try:
            data = json.loads(info.read_text())
        except Exception:
            continue
        for pt in data.get("per_task", []):
            suite = pt.get("task_group")
            if suite in out:
                out[suite].extend(bool(x) for x in pt["metrics"]["successes"])
    return out


def pc(xs):
    return 100.0 * (sum(1 for x in xs if x) / len(xs)) if xs else float("nan")


def main():
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/outputs")
    data = {mode: collect(base / d, mode) for mode, d in ARMS}

    def row(label, per_arm_lists):
        vals = {m: pc(per_arm_lists[m]) for m, _ in ARMS}
        ns = {m: len(per_arm_lists[m]) for m, _ in ARMS}
        return label, vals, ns

    rows = []
    for s in SUITES:
        rows.append(row(LABEL[s], {m: data[m][s] for m, _ in ARMS}))
    overall = {m: [v for s in SUITES for v in data[m][s]] for m, _ in ARMS}
    rows.append(row("Overall", overall))

    csv = base / "compare_ft.csv"
    with csv.open("w") as f:
        f.write("suite,teacher_pc,teacher_n,student_pc,student_n,ft_pc,ft_n,ft_vs_student,ft_vs_teacher\n")
        for label, vals, ns in rows:
            f.write(f"{label},{vals['teacher']:.1f},{ns['teacher']},"
                    f"{vals['student']:.1f},{ns['student']},"
                    f"{vals['ft']:.1f},{ns['ft']},"
                    f"{vals['ft'] - vals['student']:+.1f},{vals['ft'] - vals['teacher']:+.1f}\n")

    hdr = f"{'suite':<9} {'teacher':>9} {'distill':>9} {'finetuned':>10} {'ft-distill':>11} {'ft-teacher':>11}"
    print(hdr)
    print("-" * len(hdr))
    for label, vals, ns in rows:
        print(f"{label:<9} {vals['teacher']:>8.1f}% {vals['student']:>8.1f}% "
              f"{vals['ft']:>9.1f}% {vals['ft'] - vals['student']:>+10.1f} {vals['ft'] - vals['teacher']:>+10.1f}")
    print(f"\nwrote {csv}")


if __name__ == "__main__":
    main()
