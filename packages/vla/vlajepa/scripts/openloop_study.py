# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Extensive open-loop study: replay many LIBERO demo episodes across suites/tasks
through VLA-JEPA and aggregate per-dimension action MAE. The model (Qwen3-VL VLM +
GR00T flow-matching DiT) is loaded ONCE and reused across every episode.

Self-contained (only imports the checkpoint resolver / config repointer from
model_smoke, which is baked into the image) so it can be bind-mounted into the
container without a rebuild. Downloads use timeout+retry wrappers because the
Strix-Halo Wi-Fi link is flaky (stale-socket hangs otherwise).

Env:
  SUITES   : comma list (default libero_object,libero_spatial,libero_goal,libero_10)
  TASKS    : how many task HDF5s per suite, indexed 0..TASKS-1 (default 2)
  EPISODES : demos replayed per task file (default 8)
  MAX_STEPS: cap timesteps per episode (default: full)
  REPLAN   : re-query policy every N steps (default = action chunk)
  MODEL_REPO/CKPT_REL/BASE_VLM/BASE_ENCODER/DTYPE/OUT_DIR/GT_REPO/UNNORM_KEY/FLIP180
"""
import json
import os
import sys
import time
import csv as csvmod

import numpy as np
import torch
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_smoke import resolve_checkpoint, repoint_config  # noqa: E402

DTYPE = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[
    os.environ.get("DTYPE", "bfloat16")]
OUT_DIR = os.environ.get("OUT_DIR", "/outputs")
GT_REPO = os.environ.get("GT_REPO", "yifengzhu-hf/LIBERO-datasets")
SUITES = [s.strip() for s in os.environ.get(
    "SUITES", "libero_object,libero_spatial,libero_goal,libero_10").split(",") if s.strip()]
TASKS = int(os.environ.get("TASKS") or "2")
EPISODES = int(os.environ.get("EPISODES") or "8")
MAX_STEPS = int(os.environ["MAX_STEPS"]) if os.environ.get("MAX_STEPS") else None
FLIP180 = os.environ.get("FLIP180", "1") == "1"
DIMS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]


def hf_list_retry(repo, tries=5):
    from huggingface_hub import HfApi
    for i in range(tries):
        try:
            return HfApi().list_repo_files(repo, repo_type="dataset")
        except Exception as e:  # noqa: BLE001
            print(f"  list_repo_files retry {i+1}/{tries}: {e}", file=sys.stderr)
            time.sleep(5)
    return []


def hf_dl_retry(repo, fn, tries=5):
    from huggingface_hub import hf_hub_download
    for i in range(tries):
        try:
            return hf_hub_download(repo_id=repo, filename=fn, repo_type="dataset")
        except Exception as e:  # noqa: BLE001
            print(f"  download retry {i+1}/{tries} ({fn}): {e}", file=sys.stderr)
            time.sleep(5)
    return None


def load_demo(h5_path, episode):
    """Return (primary, wrist, state[T,8], actions[T,7], instr, n_demos) or None."""
    import h5py
    with h5py.File(h5_path, "r") as f:
        data = f["data"]
        try:
            pinfo = json.loads(data.attrs["problem_info"])
            instr = pinfo.get("language_instruction", "") or ""
            if isinstance(instr, list):
                instr = instr[0]
        except Exception:  # noqa: BLE001
            instr = ""
        demos = sorted(data.keys(), key=lambda s: int(s.split("_")[-1]))
        if episode >= len(demos):
            return None
        g = data[demos[episode]]
        obs = g["obs"]
        primary = np.asarray(obs["agentview_rgb"][:])
        wrist = np.asarray(obs["eye_in_hand_rgb"][:])
        state = np.concatenate([obs["ee_pos"][:], obs["ee_ori"][:], obs["gripper_states"][:]],
                               axis=-1).astype(np.float32)
        actions = np.asarray(g["actions"][:], dtype=np.float32)
        ndemos = len(demos)
    if not instr:
        instr = os.path.basename(h5_path).replace("_demo.hdf5", "").replace("_", " ")
    if FLIP180:
        primary = primary[:, ::-1, ::-1]
        wrist = wrist[:, ::-1, ::-1]
    return (np.ascontiguousarray(primary), np.ascontiguousarray(wrist),
            state, actions, instr, ndemos)


def unnorm(normalized, stats):
    """Upstream min/max un-normalization + corrected gripper into GT convention.

    Model emits {0,1} (1=open); LIBERO GT is {-1,+1} (+1=close) -> inverted, so
    map 1-2*g (NOT 2*g-1). Verified in the single-episode study.
    """
    lo, hi = np.asarray(stats["min"], dtype=np.float32), np.asarray(stats["max"], dtype=np.float32)
    mask = np.asarray(stats.get("mask", np.ones_like(lo, dtype=bool)))
    a = np.clip(normalized.astype(np.float32), -1, 1)
    if a.shape[-1] >= 7:
        a[:, 6] = np.where(a[:, 6] < 0.5, 0.0, 1.0)
    out = np.where(mask, 0.5 * (a + 1.0) * (hi - lo) + lo, a)
    if out.shape[-1] >= 7:
        out[:, 6] = 1.0 - 2.0 * out[:, 6]
    return out


def overlay_plot(gt, pred, mae, tag, instr, path):
    fig, axes = plt.subplots(2, 4, figsize=(17, 7), squeeze=False)
    for d in range(min(gt.shape[1], 7)):
        ax = axes[d // 4][d % 4]
        ax.plot(gt[:, d], color="tab:green", lw=1.6, label="ground truth")
        ax.plot(pred[:, d], color="tab:red", lw=1.2, ls="--", label="VLA-JEPA")
        ax.set_title(f"{DIMS[d]}  (MAE={mae[d]:.3f})", fontsize=10)
        ax.tick_params(labelsize=7)
    axes[0][0].legend(fontsize=8, loc="best")
    axes[1][3].axis("off")
    fig.suptitle(f"VLA-JEPA-LIBERO open-loop replay (ROCm gfx1151)\n{tag}  |  "
                 f"mean|MAE|={mae.mean():.3f}\n{instr}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> int:
    print(f"torch {torch.__version__} hip={torch.version.hip}")
    if not (torch.version.hip and torch.cuda.is_available()):
        print("FAIL: need a ROCm device.", file=sys.stderr)
        return 1
    print(f"device: {torch.cuda.get_device_name(0)}")
    os.makedirs(OUT_DIR, exist_ok=True)

    ckpt = resolve_checkpoint()
    repoint_config(ckpt)
    from huggingface_hub import snapshot_download
    for repo in (os.environ.get("BASE_VLM", "Qwen/Qwen3-VL-2B-Instruct"),
                 os.environ.get("BASE_ENCODER", "facebook/vjepa2-vitl-fpc64-256")):
        snapshot_download(repo_id=repo, token=os.environ.get("HF_TOKEN") or None)

    from starVLA.model.tools import read_mode_config
    from starVLA.model.framework.base_framework import baseframework
    _, norm_stats = read_mode_config(ckpt)
    unnorm_key = next(iter(norm_stats.keys())) if len(norm_stats) == 1 else \
        (os.environ.get("UNNORM_KEY") or next(iter(norm_stats.keys())))
    action_stats = norm_stats[unnorm_key]["action"]
    print(f"unnorm_key: {unnorm_key}")

    vla = baseframework.from_pretrained(ckpt).to("cuda:0").to(DTYPE).eval()
    chunk = vla.config.framework.action_model.future_action_window_size + 1
    replan = int(os.environ.get("REPLAN") or chunk)
    print(f"chunk={chunk} replan={replan}  SUITES={SUITES} TASKS={TASKS} EPISODES={EPISODES}",
          flush=True)

    rows = []
    t_start = time.time()
    for suite in SUITES:
        files = sorted(f for f in hf_list_retry(GT_REPO)
                       if f.startswith(f"{suite}/") and f.endswith(".hdf5"))
        if not files:
            print(f"WARN: no HDF5 under {suite}/ in {GT_REPO}", file=sys.stderr)
            continue
        for task_id in range(min(TASKS, len(files))):
            rel = files[task_id]
            h5 = hf_dl_retry(GT_REPO, rel)
            if not h5:
                print(f"WARN: download failed, skipping {rel}", file=sys.stderr)
                continue
            for ep in range(EPISODES):
                loaded = load_demo(h5, ep)
                if loaded is None:
                    break
                primary, wrist, state, gt_actions, instr, ndemos = loaded
                T = gt_actions.shape[0] if MAX_STEPS is None else min(MAX_STEPS, gt_actions.shape[0])
                pred = np.full((T, gt_actions.shape[1]), np.nan, dtype=np.float32)
                t = 0
                while t < T:
                    imgs = [Image.fromarray(primary[t]), Image.fromarray(wrist[t])]
                    st = state[t][None].astype(np.float32)
                    out = vla.predict_action(batch_images=[imgs], instructions=[instr], state=[st])
                    norm = np.asarray(out["normalized_actions"], dtype=np.float32)[0]
                    raw = unnorm(norm, action_stats)
                    n = min(replan, chunk, T - t)
                    pred[t:t + n] = raw[:n]
                    t += n
                if not np.isfinite(pred).all():
                    print(f"WARN: non-finite pred {suite} t{task_id} ep{ep}", file=sys.stderr)
                    continue
                gt = gt_actions[:T]
                mae = np.mean(np.abs(pred - gt), axis=0)
                row = {"suite": suite, "task_id": task_id, "file": os.path.basename(rel),
                       "episode": ep, "T": int(T), "instr": instr}
                for i, dn in enumerate(DIMS):
                    row[f"mae_{dn}"] = float(mae[i])
                row["mae_mean"] = float(mae.mean())
                rows.append(row)
                el = time.time() - t_start
                print(f"[{el:6.0f}s] {suite} t{task_id} ep{ep}: T={T} "
                      f"mean|MAE|={mae.mean():.3f} grip={mae[6]:.3f}", flush=True)
                if ep == 0:
                    overlay_plot(gt, pred, mae, f"{suite}_task{task_id}_ep0", instr,
                                 os.path.join(OUT_DIR, f"study_example_{suite}_task{task_id}.png"))

    if not rows:
        print("FAIL: no episodes completed.", file=sys.stderr)
        return 1

    keys = list(rows[0].keys())
    with open(os.path.join(OUT_DIR, "study_results.csv"), "w", newline="") as fp:
        w = csvmod.DictWriter(fp, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    json.dump(rows, open(os.path.join(OUT_DIR, "study_results.json"), "w"), indent=2)

    arr = np.array([[r[f"mae_{d}"] for d in DIMS] for r in rows])  # [N, 7]
    suites_of = np.array([r["suite"] for r in rows])
    uniq = list(dict.fromkeys(suites_of.tolist()))
    N = len(rows)
    print(f"\n==== STUDY SUMMARY: {N} episodes across {len(uniq)} suite(s) ====")
    print("overall per-dim MAE: " + ", ".join(f"{d}={arr[:, i].mean():.3f}"
                                               for i, d in enumerate(DIMS)))
    print(f"overall mean|MAE| = {arr.mean():.3f}")
    for s in uniq:
        m = arr[suites_of == s]
        print(f"  {s:16s} n={len(m):3d}  mean|MAE|={m.mean():.3f}  gripper={m[:, 6].mean():.3f}")

    # Plot 1: grouped bar, per-dim MAE by suite (mean +/- std).
    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(7)
    wbar = 0.8 / max(len(uniq), 1)
    for j, s in enumerate(uniq):
        m = arr[suites_of == s]
        ax.bar(x + j * wbar, m.mean(0), wbar, yerr=m.std(0), capsize=2, label=f"{s} (n={len(m)})")
    ax.set_xticks(x + 0.4 - wbar / 2)
    ax.set_xticklabels(DIMS)
    ax.set_ylabel("per-dim action MAE")
    ax.set_title(f"VLA-JEPA open-loop action MAE by dimension & suite (ROCm gfx1151)\n"
                 f"{N} episodes | VLA-JEPA-LIBERO | overall mean|MAE|={arr.mean():.3f}")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "study_mae_by_dim_suite.png"), dpi=120)
    plt.close(fig)

    # Plot 2: distribution of episode mean|MAE| by suite (boxplot).
    fig, ax = plt.subplots(figsize=(10, 5))
    data = [arr[suites_of == s].mean(1) for s in uniq]
    ax.boxplot(data, showmeans=True)
    ax.set_xticks(range(1, len(uniq) + 1))
    ax.set_xticklabels([f"{s}\n(n={len(d)})" for s, d in zip(uniq, data)])
    ax.set_ylabel("episode mean|MAE| (7-DoF)")
    ax.set_title(f"VLA-JEPA open-loop tracking-error distribution by suite\n{N} episodes")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "study_mae_distribution.png"), dpi=120)
    plt.close(fig)

    print("PASS: VLA-JEPA open-loop study complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
