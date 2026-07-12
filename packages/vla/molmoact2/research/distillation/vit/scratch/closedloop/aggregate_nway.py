"""N-way closed-loop LIBERO comparison across the distillation experiments.

Arms (included only if their result dir exists):
  teacher          stock MolmoAct2 SigLIP2 ViT              vd_eval_teacher   (mode teacher)
  distill_hybrid   distill-only hybrid student (no ft)      vd_eval_student   (mode student)
  ft_hybrid        finetuned hybrid (original baseline)     vd_eval_ft        (mode ft)
  ft_siglip_nano   finetuned attention-only tinyvit         vd_eval_ft_siglip_nano
  ft_hybrid_droid  finetuned hybrid + DROID real data       vd_eval_ft_hybrid_droid
  ft_cnn_fpga      finetuned pure-CNN (FPGA-friendly)       vd_eval_ft_cnn_fpga

Reads eval_info.json per (suite,task) unit, computes per-suite + overall
pc_success, and joins static FLOPs/params/compression metadata. Stdlib-only.

Usage: python aggregate_nway.py [BASE_DIR]   (default /outputs)
Writes BASE_DIR/compare_nway.csv and prints a table.
"""

import json
import sys
from pathlib import Path

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
LABEL = {"libero_spatial": "Spatial", "libero_object": "Object",
         "libero_goal": "Goal", "libero_10": "Long"}

# (arm_key, result_dir, unit_mode)
ARMS = [
    ("teacher",         "vd_eval_teacher",          "teacher"),
    ("distill_hybrid",  "vd_eval_student",          "student"),
    ("ft_hybrid",       "vd_eval_ft",               "ft"),
    ("ft_siglip_nano",  "vd_eval_ft_siglip_nano",   "ft"),
    ("ft_hybrid_droid", "vd_eval_ft_hybrid_droid",  "ft"),
    ("ft_cnn_fpga",     "vd_eval_ft_cnn_fpga",      "ft"),
    ("run_d_joint",     "vd_eval_joint",            "joint"),
]

# static architecture metadata (from distill.flops; teacher ~617 GFLOPs/crop)
META = {
    "teacher":         dict(params_m=None, gflops=617.15, comp=1.0,   arch="SigLIP2 ViT (frozen)"),
    "distill_hybrid":  dict(params_m=2.79, gflops=5.40,   comp=114.3, arch="conv+attn hybrid"),
    "ft_hybrid":       dict(params_m=2.79, gflops=5.40,   comp=114.3, arch="conv+attn hybrid"),
    "ft_siglip_nano":  dict(params_m=2.73, gflops=5.74,   comp=107.5, arch="attention-only (tinyvit)"),
    "ft_hybrid_droid": dict(params_m=2.79, gflops=5.40,   comp=114.3, arch="hybrid + DROID real data"),
    "ft_cnn_fpga":     dict(params_m=3.83, gflops=5.14,   comp=120.1, arch="pure-CNN (FPGA-friendly)"),
    # Run D also compresses the LLM: thin-twin student LLM (~26M) ~112x vs teacher
    # text transformer; gflops/comp columns below stay ViT-per-crop for comparability.
    "run_d_joint":     dict(params_m=2.79, gflops=5.40,   comp=114.3, arch="hybrid_droid ViT + thin-twin LLM +LoRA"),
}


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
    present = [(k, d, m) for k, d, m in ARMS if (base / d).exists()]
    if not present:
        print("no eval result dirs found under", base)
        return
    data = {k: collect(base / d, m) for k, d, m in present}

    csv = base / "compare_nway.csv"
    keys = [k for k, _, _ in present]
    with csv.open("w") as f:
        f.write("suite," + ",".join(f"{k}_pc,{k}_n" for k in keys) + "\n")
        for s in SUITES:
            cells = [f"{pc(data[k][s]):.1f},{len(data[k][s])}" for k in keys]
            f.write(f"{LABEL[s]}," + ",".join(cells) + "\n")
        ov = {k: [v for s in SUITES for v in data[k][s]] for k in keys}
        cells = [f"{pc(ov[k]):.1f},{len(ov[k])}" for k in keys]
        f.write("Overall," + ",".join(cells) + "\n")
        f.write("\n# arch metadata\narm,arch,params_M,gflops_per_crop,compression_x,overall_pc\n")
        for k in keys:
            mt = META.get(k, {})
            pm = "" if mt.get("params_m") is None else f"{mt['params_m']:.2f}"
            f.write(f"{k},{mt.get('arch','')},{pm},{mt.get('gflops','')},"
                    f"{mt.get('comp','')},{pc(ov[k]):.1f}\n")

    # printed success table
    w = 16
    hdr = f"{'suite':<9}" + "".join(f"{k:>{w}}" for k in keys)
    print(hdr); print("-" * len(hdr))
    for s in SUITES:
        print(f"{LABEL[s]:<9}" + "".join(f"{pc(data[k][s]):>{w-1}.1f}%" for k in keys))
    print(f"{'Overall':<9}" + "".join(f"{pc(ov[k]):>{w-1}.1f}%" for k in keys))
    # arch/FLOPs table
    print()
    print(f"{'arm':<16}{'arch':<28}{'params(M)':>10}{'GFLOPs':>9}{'comp':>8}{'overall':>9}")
    print("-" * 80)
    for k in keys:
        mt = META.get(k, {})
        pm = "-" if mt.get("params_m") is None else f"{mt['params_m']:.2f}"
        gf = mt.get("gflops", "")
        cp = mt.get("comp", "")
        print(f"{k:<16}{mt.get('arch',''):<28}{pm:>10}{gf:>9}{('%.0fx'%cp) if cp else '':>8}{pc(ov[k]):>8.1f}%")
    print(f"\nwrote {csv}")


if __name__ == "__main__":
    main()
