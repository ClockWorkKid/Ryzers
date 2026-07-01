# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Plot the RT smoothness ablation (sync / async-hold / async-blend).

Reads the per-episode action streams (<suite>_t<task>_<mode>_actions.npy) and the
summary CSV written by rt_smoothness_ablation.py, and produces small PNGs:

  - speed_<suite>_t<task>.png : commanded end-effector speed (||motion delta|| per tick)
    for the three modes overlaid on shared axes. There is no single ground-truth motion
    here (each regime produces a different trajectory), so per workspace rule 2.a we
    overlay the modes for comparison; the pauses show up as speed dropping to ~0.
  - jerk_<suite>_t<task>.png  : commanded jerk magnitude per tick, modes overlaid.
  - summary_bars.png          : success / hold% / throughput / mean|accel| / max|jerk|
    bar charts aggregated per mode.

Outputs are intentionally small (downscaled, dpi=110) so they can live under artifacts/.

Usage (in a venv with matplotlib + numpy):
  python plot_rt_smoothness.py --in /outputs/rt_smoothness --out /outputs/rt_smoothness/plots
"""
import argparse
import csv
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

MODES = ["sync", "hold", "blend"]
COLORS = {"sync": "#ef4444", "hold": "#f59e0b", "blend": "#22c55e"}
LABELS = {"sync": "sync (stop-and-decide)", "hold": "async-hold", "blend": "async-blend (RTC)"}


def _speed(actions):
    """Per-tick commanded speed = ||motion delta|| (first 3 dims dominate translation)."""
    a = np.asarray(actions, dtype=np.float64)
    return np.linalg.norm(a[:, :6], axis=1)


def _jerk(actions):
    a = np.asarray(actions, dtype=np.float64)[:, :6]
    accel = np.diff(a, axis=0)
    jerk = np.diff(accel, axis=0)
    return np.linalg.norm(jerk, axis=1)


def _episode_key(path):
    base = os.path.basename(path)[: -len("_actions.npy")]
    mode = base.rsplit("_", 1)[-1]
    stem = base[: -(len(mode) + 1)]
    return stem, mode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", default="/outputs/rt_smoothness")
    ap.add_argument("--out", dest="outdir", default="/outputs/rt_smoothness/plots")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # group action streams by episode stem (suite_t<task>) -> {mode: actions}
    episodes = {}
    for p in sorted(glob.glob(os.path.join(args.indir, "*_actions.npy"))):
        stem, mode = _episode_key(p)
        episodes.setdefault(stem, {})[mode] = np.load(p)

    for stem, by_mode in episodes.items():
        # speed overlay
        fig, ax = plt.subplots(figsize=(7, 3))
        for mode in MODES:
            if mode not in by_mode:
                continue
            sp = _speed(by_mode[mode])
            ax.plot(np.arange(len(sp)), sp, color=COLORS[mode], lw=1.2, label=LABELS[mode])
        ax.set_title(f"Commanded speed per control tick - {stem}")
        ax.set_xlabel("control tick")
        ax.set_ylabel("||motion delta||")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, f"speed_{stem}.png"), dpi=110)
        plt.close(fig)

        # jerk overlay
        fig, ax = plt.subplots(figsize=(7, 3))
        for mode in MODES:
            if mode not in by_mode:
                continue
            jk = _jerk(by_mode[mode])
            ax.plot(np.arange(len(jk)), jk, color=COLORS[mode], lw=1.0, label=LABELS[mode])
        ax.set_title(f"Commanded jerk magnitude per tick - {stem}")
        ax.set_xlabel("control tick")
        ax.set_ylabel("||jerk||")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, f"jerk_{stem}.png"), dpi=110)
        plt.close(fig)

    # summary bars from the newest CSV
    csvs = sorted(glob.glob(os.path.join(args.indir, "summary_*.csv")))
    if csvs:
        rows = list(csv.DictReader(open(csvs[-1])))
        metrics = [("success", "success rate"), ("hold_pct", "hold / stall %"),
                   ("throughput_sps", "throughput (steps/s)"),
                   ("mean_accel", "mean |accel|"), ("max_jerk", "max |jerk|")]
        present = [m for m in MODES if any(r["mode"] == m for r in rows)]

        def agg(mode, key):
            vals = [float(r[key]) for r in rows if r["mode"] == mode]
            return sum(vals) / len(vals) if vals else 0.0

        fig, axes = plt.subplots(1, len(metrics), figsize=(3.0 * len(metrics), 3.2))
        for ax, (key, title) in zip(axes, metrics):
            vals = [agg(m, key) for m in present]
            ax.bar(present, vals, color=[COLORS[m] for m in present])
            ax.set_title(title, fontsize=10)
            ax.grid(alpha=0.2, axis="y")
            for i, v in enumerate(vals):
                ax.text(i, v, f"{v:.3g}", ha="center", va="bottom", fontsize=8)
        fig.suptitle("RT chunk-stitching ablation (mean over tasks)", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(os.path.join(args.outdir, "summary_bars.png"), dpi=110)
        plt.close(fig)
        print("wrote summary_bars.png from", os.path.basename(csvs[-1]))

    print("plots ->", args.outdir)


if __name__ == "__main__":
    main()
