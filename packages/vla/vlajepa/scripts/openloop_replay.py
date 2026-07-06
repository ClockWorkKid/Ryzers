# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Capability 2: open-loop replay of a real LIBERO demo episode on ROCm.

For a ground-truth LIBERO demonstration (recorded observations + 7-DoF actions),
we feed the *recorded* observations to VLA-JEPA's ``predict_action`` and compare the
model's predicted action trajectory against the demonstrator's ground-truth actions.
This is OPEN-LOOP: we never step a simulator, so there is no compounding error and
no MuJoCo dependency -- we only need to read the demo HDF5.

Observation / state / un-normalization follow the upstream LIBERO deployment adapter
(examples/LIBERO/model2libero_interface.py + eval_libero.py):
  * two views: agentview (primary) + eye-in-hand (wrist), rotated 180 deg to match
    the training preprocessing, resized to the model's input size.
  * state (8-d): [eef_pos(3), quat->axisangle(3), gripper_qpos(2)]  (already stored
    in the demo obs; we read it directly).
  * predict_action -> normalized_actions [B, chunk, 7]; un-normalized with
    dataset_statistics.json[unnorm_key]["action"] (min/max/mask), gripper binarized.

Per workspace rule 2.a (numeric data with GT): predicted and ground-truth actions are
overlaid on the same per-dimension plots for direct comparison.

Env (see config.yaml):
  MODEL_REPO, CKPT_REL, BASE_VLM, BASE_ENCODER, DTYPE, OUT_DIR
  GT_REPO   : HF dataset repo with LIBERO demo HDF5 (default yifengzhu-hf/LIBERO-datasets)
  SUITE     : libero_spatial|libero_object|libero_goal|libero_10|libero_90
  GT_FILE   : explicit "<suite>/<name>.hdf5" in the repo (overrides SUITE/TASK_ID pick)
  GT_LOCAL  : path to a local .hdf5 (skips download entirely)
  TASK_ID   : index into the sorted suite file list when GT_FILE unset (default 0)
  EPISODE   : demo index within the HDF5 (default 0)
  INSTRUCTION : override task language (else read from HDF5 problem_info / filename)
  REPLAN    : re-query the policy every N steps (default = action chunk size)
  MAX_STEPS : cap replayed timesteps (default: full episode)
  FLIP180   : 1/0, rotate views 180 deg to match training (default 1)
"""
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Reuse the smoke's checkpoint resolver + config repointer (same /ryzers dir).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_smoke import resolve_checkpoint, repoint_config  # noqa: E402

MODEL_REPO = os.environ.get("MODEL_REPO", "ginwind/VLA-JEPA")
CKPT_REL = os.environ.get("CKPT_REL", "LIBERO/checkpoints/VLA-JEPA-LIBERO.pt")
DTYPE = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[
    os.environ.get("DTYPE", "bfloat16")]
OUT_DIR = os.environ.get("OUT_DIR", "/outputs")

GT_REPO = os.environ.get("GT_REPO", "yifengzhu-hf/LIBERO-datasets")
SUITE = os.environ.get("SUITE") or "libero_object"
GT_FILE = os.environ.get("GT_FILE") or ""
GT_LOCAL = os.environ.get("GT_LOCAL") or ""
TASK_ID = int(os.environ.get("TASK_ID") or "0")
EPISODE = int(os.environ.get("EPISODE") or "0")
INSTRUCTION_OVERRIDE = os.environ.get("INSTRUCTION") or ""
MAX_STEPS = int(os.environ["MAX_STEPS"]) if os.environ.get("MAX_STEPS") else None
FLIP180 = os.environ.get("FLIP180", "1") == "1"

DIMS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]


def resolve_gt_hdf5() -> str:
    """Return a local path to a single LIBERO demo HDF5 (download one file if needed)."""
    if GT_LOCAL:
        if not os.path.isfile(GT_LOCAL):
            print(f"FAIL: GT_LOCAL not found: {GT_LOCAL}", file=sys.stderr)
            sys.exit(1)
        return GT_LOCAL

    from huggingface_hub import HfApi, hf_hub_download

    rel = GT_FILE
    if not rel:
        files = HfApi().list_repo_files(GT_REPO, repo_type="dataset")
        suite_files = sorted(
            f for f in files if f.startswith(f"{SUITE}/") and f.endswith(".hdf5"))
        if not suite_files:
            print(f"FAIL: no .hdf5 under {SUITE}/ in {GT_REPO}", file=sys.stderr)
            sys.exit(1)
        if TASK_ID >= len(suite_files):
            print(f"FAIL: TASK_ID {TASK_ID} >= {len(suite_files)} files in {SUITE}/",
                  file=sys.stderr)
            sys.exit(1)
        rel = suite_files[TASK_ID]
        print(f"gt file        : [{TASK_ID}/{len(suite_files)}] {rel}")
    print(f"downloading GT : {GT_REPO}::{rel} (single file)")
    return hf_hub_download(repo_id=GT_REPO, filename=rel, repo_type="dataset")


def load_episode(h5_path: str):
    """Load one LIBERO demo: (primary[T,H,W,3], wrist[T,H,W,3], state[T,8], actions[T,7], instr)."""
    import h5py

    with h5py.File(h5_path, "r") as f:
        data = f["data"]
        instr = INSTRUCTION_OVERRIDE
        if not instr:
            try:
                pinfo = json.loads(data.attrs["problem_info"])
                instr = pinfo.get("language_instruction", "") or ""
                if isinstance(instr, list):
                    instr = instr[0]
            except Exception:  # noqa: BLE001
                instr = ""
        demos = sorted(data.keys(), key=lambda s: int(s.split("_")[-1]))
        if EPISODE >= len(demos):
            print(f"FAIL: EPISODE {EPISODE} >= {len(demos)} demos", file=sys.stderr)
            sys.exit(1)
        g = data[demos[EPISODE]]
        obs = g["obs"]
        primary = np.asarray(obs["agentview_rgb"][:])
        wrist = np.asarray(obs["eye_in_hand_rgb"][:])
        ee_pos = np.asarray(obs["ee_pos"][:])
        ee_ori = np.asarray(obs["ee_ori"][:])
        grip = np.asarray(obs["gripper_states"][:])
        actions = np.asarray(g["actions"][:], dtype=np.float32)
        state = np.concatenate([ee_pos, ee_ori, grip], axis=-1).astype(np.float32)

    if not instr:
        base = os.path.basename(h5_path)
        instr = base.replace("_demo.hdf5", "").replace("_", " ")
        print(f"WARN: no language in HDF5; using filename-derived instruction",
              file=sys.stderr)
    if FLIP180:
        primary = primary[:, ::-1, ::-1]
        wrist = wrist[:, ::-1, ::-1]
    return (np.ascontiguousarray(primary), np.ascontiguousarray(wrist),
            state, actions, instr)


def unnormalize(normalized: np.ndarray, stats: dict) -> np.ndarray:
    """Inverse of the training min/max normalization (matches model2libero_interface)."""
    lo, hi = np.asarray(stats["min"], dtype=np.float32), np.asarray(stats["max"], dtype=np.float32)
    mask = np.asarray(stats.get("mask", np.ones_like(lo, dtype=bool)))
    a = np.clip(normalized.astype(np.float32), -1, 1)
    if a.shape[-1] >= 7:
        a[:, 6] = np.where(a[:, 6] < 0.5, 0.0, 1.0)  # binarize gripper (open=1)
    return np.where(mask, 0.5 * (a + 1.0) * (hi - lo) + lo, a)


def main() -> int:
    if os.environ.get("FETCH_ONLY") == "1":
        # Prefetch the GT episode file only (no GPU / model load) for download scripts.
        h5 = resolve_gt_hdf5()
        print(f"gt cached      : {h5}")
        print("PASS: GT episode fetched")
        return 0

    print(f"torch          : {torch.__version__} hip={torch.version.hip}")
    if not (torch.version.hip and torch.cuda.is_available()):
        print("FAIL: need a ROCm device.", file=sys.stderr)
        return 1
    print(f"device[0]      : {torch.cuda.get_device_name(0)}")
    os.makedirs(OUT_DIR, exist_ok=True)

    ckpt = resolve_checkpoint()
    repoint_config(ckpt)

    # Warm base models into cache (predict path constructs them on load).
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
    print(f"unnorm_key     : {unnorm_key}  (dims={len(action_stats['min'])})")

    vla = baseframework.from_pretrained(ckpt).to("cuda:0").to(DTYPE).eval()
    chunk = vla.config.framework.action_model.future_action_window_size + 1
    replan = int(os.environ.get("REPLAN") or chunk)
    print(f"chunk={chunk}  replan={replan}")

    h5 = resolve_gt_hdf5()
    primary, wrist, state, gt_actions, instr = load_episode(h5)
    T = gt_actions.shape[0] if MAX_STEPS is None else min(MAX_STEPS, gt_actions.shape[0])
    print(f"episode        : T={T}  views={primary.shape} state={state.shape}")
    print(f"instruction    : {instr}")

    pred_raw = np.full((T, gt_actions.shape[1]), np.nan, dtype=np.float32)
    t = 0
    while t < T:
        imgs = [Image.fromarray(primary[t]), Image.fromarray(wrist[t])]
        st = state[t][None].astype(np.float32)  # (1, 8)
        out = vla.predict_action(batch_images=[imgs], instructions=[instr], state=[st])
        norm = np.asarray(out["normalized_actions"], dtype=np.float32)[0]  # (chunk, 7)
        raw = unnormalize(norm, action_stats)
        n = min(replan, chunk, T - t)
        pred_raw[t:t + n] = raw[:n]
        t += n
    if not np.isfinite(pred_raw).all():
        print("FAIL: non-finite / unfilled predicted actions.", file=sys.stderr)
        return 1

    # Gripper convention reconciliation. Upstream un-normalization (mirrored above)
    # emits {0, 1} with 1=open / 0=close; the LIBERO demo GT stores the robosuite
    # action in {-1, +1} with +1=close. These are INVERTED, so the correct map into
    # the GT convention is 1-2*g (open: 1->-1, close: 0->+1), NOT 2*g-1. Verified
    # against the raw arrays (per-third phase means align under 1-2*g; gripper MAE
    # drops from 1.37 to ~0.64, residual being transition-timing, not sign).
    if pred_raw.shape[1] >= 7:
        pred_raw[:, 6] = 1.0 - 2.0 * pred_raw[:, 6]

    gt = gt_actions[:T]
    mae = np.mean(np.abs(pred_raw - gt), axis=0)
    tag = f"{SUITE}_task{TASK_ID}_ep{EPISODE}"

    # Overlay plot: GT vs prediction on the same axes, per action dim (rule 2.a).
    fig, axes = plt.subplots(2, 4, figsize=(17, 7), squeeze=False)
    for d in range(min(gt.shape[1], 7)):
        ax = axes[d // 4][d % 4]
        ax.plot(gt[:, d], color="tab:green", lw=1.6, label="ground truth")
        ax.plot(pred_raw[:, d], color="tab:red", lw=1.2, ls="--", label="VLA-JEPA")
        ax.set_title(f"{DIMS[d]}  (MAE={mae[d]:.3f})", fontsize=10)
        ax.tick_params(labelsize=7)
    axes[0][0].legend(fontsize=8, loc="best")
    axes[1][3].axis("off")
    fig.suptitle(f"VLA-JEPA-LIBERO open-loop replay (ROCm gfx1151)\n"
                 f"{tag}  replan={replan}  |  mean|MAE|={mae.mean():.3f}\n{instr}",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    png = os.path.join(OUT_DIR, f"openloop_{tag}_actions.png")
    fig.savefig(png, dpi=110)
    plt.close(fig)

    # Subsampled frame strip (primary view) for context.
    idx = np.linspace(0, T - 1, min(8, T)).astype(int)
    strip = np.concatenate([primary[i] for i in idx], axis=1)
    Image.fromarray(strip).save(os.path.join(OUT_DIR, f"openloop_{tag}_frames.png"))
    np.savez(os.path.join(OUT_DIR, f"openloop_{tag}.npz"),
             pred=pred_raw, gt=gt, state=state[:T], mae=mae)

    print(f"per-dim MAE    : " + ", ".join(f"{DIMS[d]}={mae[d]:.3f}" for d in range(gt.shape[1])))
    print(f"action plot    : {png}")
    print("PASS: VLA-JEPA open-loop replay OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
