# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Phase 3 - open-loop DROID evaluation for Cosmos3-Nano-Policy-DROID on ROCm (gfx1151).

Replays real Cosmos3-DROID episodes (LeRobotDataset v3.0) through the released policy and scores
the predicted action chunk against the dataset's ground-truth actions:

  for each queried frame t in an episode:
    obs  = { 3 raw camera frames @ t, joint_positions[t] (7), gripper_position[t] (1), task }
    pred = RobolabPolicyService.infer(obs)["action"]          # [chunk,8] raw joint_pos space
    gt   = [action.joint_position[t:t+chunk], action.gripper_position[t:t+chunk]]  # [chunk,8]

The server composes the exact training concat view from the 3 raw cameras (wrist on top; left|right
half-size below -> 540x640) via _compose_roboarena_views, flips the input gripper internally, and
un-flips the returned gripper back to raw units -> pred and gt are directly comparable in raw
joint-position space (rad for the 7 joints, [0,1] for the gripper). We reuse the upstream server +
dataset contracts verbatim (rule 2.1); only the ROCm SDPA attention patch + guardrails-off shim are
applied, exactly as in the Phase 2 smoke.

Reports per-dimension RMSE/MAE (raw + std-normalized), RMSE growth over the 32-step horizon, and
per-call latency. Saves GT-vs-pred overlays (rule 2.a: GT and prediction on the same axes) and a
metrics.json to OUT_DIR. Optional world-model rollout video (DECODE_VIDEO=1) is handled by the
companion two-column renderer once VAE decode is fast enough (Phase 2b).

Run (inside the cosmos3 image, GPU passthrough), e.g.:
    DROID_ROOT=/models/cosmos3_droid/success NUM_EPISODES=10 QUERIES_PER_EP=8 \
      CKPT=/models/Cosmos3-Nano-Policy-DROID python /work/openloop_replay.py
"""
import glob
import json
import os
import sys
import time

import numpy as np
import pyarrow.parquet as pq

FPS = 15.0
CAMS = {
    "wrist": "observation.image.wrist_image_left",
    "left": "observation.image.exterior_image_1_left",
    "right": "observation.image.exterior_image_2_left",
}
# RoBoArena obs keys the server composes into the concat view (see _compose_roboarena_views).
OBS_KEY = {
    "wrist": "observation/wrist_image_left",
    "left": "observation/exterior_image_1_left",
    "right": "observation/exterior_image_2_left",
}
JOINT_ACT = "action.joint_position"
GRIP_ACT = "action.gripper_position"
JOINT_STATE = "observation.state.joint_positions"
GRIP_STATE = "observation.state.gripper_position"
DIM_LABELS = [f"joint{i}" for i in range(7)] + ["gripper"]


def _read_episodes_meta(root: str) -> "list[dict]":
    """Per-episode records with global row span, task, and per-camera video locators."""
    files = sorted(glob.glob(os.path.join(root, "meta", "episodes", "chunk-*", "file-*.parquet")))
    if not files:
        raise FileNotFoundError(f"no episodes meta under {root}/meta/episodes")
    df = pq.read_table(files[0]).to_pandas()
    recs = []
    for _, r in df.iterrows():
        rec = {
            "episode_index": int(r["episode_index"]),
            "length": int(r["length"]),
            "from_index": int(r["dataset_from_index"]),
            "to_index": int(r["dataset_to_index"]),
            "tasks": r.get("tasks"),
        }
        ok = True
        for name, key in CAMS.items():
            fi = r.get(f"videos/{key}/file_index")
            ci = r.get(f"videos/{key}/chunk_index")
            ft = r.get(f"videos/{key}/from_timestamp")
            if fi is None or ft is None:
                ok = False
                break
            rec[f"{name}_file_index"] = int(fi)
            rec[f"{name}_chunk_index"] = int(ci) if ci is not None else 0
            rec[f"{name}_from_ts"] = float(ft)
        rec["all_cams_local"] = ok and all(rec.get(f"{n}_file_index") == 0 and rec.get(f"{n}_chunk_index") == 0 for n in CAMS)
        recs.append(rec)
    return recs


def _task_string(rec: dict, task_map: dict, data_slice) -> str:
    """First annotation of the multi-language task label ('a | b | c' -> 'a')."""
    t = rec.get("tasks")
    if isinstance(t, (list, np.ndarray)) and len(t):
        t = t[0]
    if isinstance(t, str) and t:
        return t.split(" | ")[0]
    ti = int(np.asarray(data_slice["task_index"])[0])
    s = task_map.get(ti, "complete the manipulation task")
    return s.split(" | ")[0]


def _open_video(root: str, cam_key: str):
    import av
    p = sorted(glob.glob(os.path.join(root, "videos", cam_key, "chunk-000", "file-000.mp4")))
    if not p:
        raise FileNotFoundError(f"missing video shard for {cam_key}")
    return av.open(p[0])


def _frame_at(container, target_s: float) -> np.ndarray:
    """Decode the frame at/just after target_s (keyframe-backed seek). Returns [H,W,3] uint8 RGB."""
    stream = container.streams.video[0]
    target_pts = int(target_s / stream.time_base)
    container.seek(max(target_pts, 0), backward=True, any_frame=False, stream=stream)
    best = None
    for frame in container.decode(stream):
        if frame.pts is None:
            best = frame
            continue
        best = frame
        if frame.pts >= target_pts:
            break
    if best is None:
        raise RuntimeError("no frame decoded")
    return best.to_ndarray(format="rgb24")


def _load_std(root: str):
    """Per-dim std for the 8-dim joint_pos action, from meta/stats.json (for normalized RMSE)."""
    p = os.path.join(root, "meta", "stats.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        s = json.load(f)
    try:
        js = np.asarray(s[JOINT_ACT]["std"], dtype=np.float64).reshape(-1)[:7]
        gs = np.asarray(s[GRIP_ACT]["std"], dtype=np.float64).reshape(-1)[:1]
        std = np.concatenate([js, gs])
        std[std < 1e-6] = 1.0
        return std
    except Exception as exc:
        print(f"WARN: could not parse action std: {exc!r}", file=sys.stderr)
        return None


def _plots(out_dir: str, per_dim_rmse, per_dim_rmse_n, horizon_rmse, sample):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"WARN: matplotlib unavailable, skipping plots: {exc!r}", file=sys.stderr)
        return []
    saved = []

    # 1) per-dim RMSE (raw + normalized)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].bar(DIM_LABELS, per_dim_rmse, color="#c1121f")
    ax[0].set_title("Open-loop action RMSE (raw units)")
    ax[0].set_ylabel("RMSE (rad / gripper)")
    ax[0].tick_params(axis="x", rotation=45)
    if per_dim_rmse_n is not None:
        ax[1].bar(DIM_LABELS, per_dim_rmse_n, color="#003049")
        ax[1].set_title("Open-loop action RMSE (std-normalized)")
        ax[1].set_ylabel("RMSE / std")
        ax[1].tick_params(axis="x", rotation=45)
    fig.tight_layout()
    f1 = os.path.join(out_dir, "openloop_rmse_per_dim.png")
    fig.savefig(f1, dpi=110)
    plt.close(fig)
    saved.append(f1)

    # 2) RMSE growth over horizon
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(1, len(horizon_rmse) + 1), horizon_rmse, "-o", ms=3, color="#c1121f")
    ax.set_xlabel("prediction horizon step")
    ax.set_ylabel("mean RMSE over 8 dims (raw)")
    ax.set_title("Action error vs horizon step (open-loop)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    f2 = os.path.join(out_dir, "openloop_rmse_vs_horizon.png")
    fig.savefig(f2, dpi=110)
    plt.close(fig)
    saved.append(f2)

    # 3) GT vs pred overlay for a sample (ep,query): 8 dims, GT solid vs pred dashed (rule 2.a)
    if sample is not None:
        gt, pred, meta = sample
        h = np.arange(gt.shape[0])
        fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True)
        for d, a in enumerate(axes.ravel()):
            a.plot(h, gt[:, d], "-", color="#003049", label="GT")
            a.plot(h, pred[:, d], "--", color="#c1121f", label="pred")
            a.set_title(DIM_LABELS[d])
            a.grid(alpha=0.3)
            if d == 0:
                a.legend(fontsize=8)
        fig.suptitle(f"GT vs predicted action chunk - ep {meta['ep']} @ frame {meta['t']}: \"{meta['task'][:70]}\"")
        fig.tight_layout()
        f3 = os.path.join(out_dir, "openloop_gt_vs_pred_sample.png")
        fig.savefig(f3, dpi=110)
        plt.close(fig)
        saved.append(f3)
    return saved


def main() -> int:
    import torch

    root = os.environ.get("DROID_ROOT", "/models/cosmos3_droid/success")
    ckpt = os.environ.get("CKPT", "/models/Cosmos3-Nano-Policy-DROID")
    n_ep = int(os.environ.get("NUM_EPISODES", "10"))
    q_per_ep = int(os.environ.get("QUERIES_PER_EP", "8"))
    num_steps = int(os.environ.get("NUM_STEPS", "4"))
    out_dir = os.environ.get("OUT_DIR", "/outputs")
    seed = int(os.environ.get("SEED", "0"))
    os.makedirs(out_dir, exist_ok=True)

    print(f"DROID_ROOT   : {root}", flush=True)
    print(f"checkpoint   : {ckpt}", flush=True)
    print(f"episodes     : {n_ep}   queries/ep: {q_per_ep}   num_steps: {num_steps}", flush=True)

    if not torch.cuda.is_available():
        print("FAIL: no ROCm device visible.", file=sys.stderr)
        return 1

    # --- dataset ---------------------------------------------------------------
    ep_recs = _read_episodes_meta(root)
    usable = [r for r in ep_recs if r["all_cams_local"]]
    usable.sort(key=lambda r: r["episode_index"])
    if not usable:
        print("FAIL: no episodes have all 3 camera shards locally (need file_index==0).", file=sys.stderr)
        return 2
    sel = usable[:n_ep]
    print(f"usable episodes (all cams local): {len(usable)}  -> using {len(sel)}", flush=True)

    data_files = sorted(glob.glob(os.path.join(root, "data", "chunk-*", "file-*.parquet")))
    cols = [JOINT_STATE, GRIP_STATE, JOINT_ACT, GRIP_ACT, "frame_index", "episode_index", "index", "task_index"]
    ddf = pq.read_table(data_files[0], columns=cols).to_pandas()
    tasks_df = pq.read_table(os.path.join(root, "meta", "tasks.parquet")).to_pandas()
    task_map = {int(v): str(k) for k, v in zip(tasks_df.index, tasks_df["task_index"])}
    action_std = _load_std(root)

    # --- policy (same enablement as Phase 2 smoke) -----------------------------
    from cosmos_framework.scripts import action_policy_server_robolab as srv
    from cosmos_framework.scripts.action_policy_server_robolab import RobolabPolicyService, RobolabServerArgs

    _orig = RobolabPolicyService._build_setup_args

    def _no_guardrails(self, a):
        s = _orig(self, a)
        try:
            s.guardrails = False
        except Exception:
            pass
        return s

    srv.RobolabPolicyService._build_setup_args = _no_guardrails

    import cosmos3_rocm_patches
    cosmos3_rocm_patches.apply()

    t0 = time.time()
    svc = RobolabPolicyService(
        RobolabServerArgs(
            checkpoint_path=ckpt, decode_video=False, num_steps=num_steps, deterministic_seed=True, seed=seed
        )
    )
    chunk = int(svc.cfg.action_chunk_size)
    adim = int(svc.cfg.action_dim)
    print(f"policy ready : load={time.time() - t0:.1f}s  chunk={chunk} action_dim={adim}", flush=True)

    # --- eval loop -------------------------------------------------------------
    sq_err = np.zeros((chunk, adim), dtype=np.float64)  # summed sq error per (horizon, dim)
    abs_err = np.zeros((chunk, adim), dtype=np.float64)
    n_terms = 0
    latencies = []
    sample_for_plot = None
    best_motion = -1.0  # pick the highest-motion query for the GT-vs-pred overlay (rule 2.a)
    cache_gt, cache_pred, cache_meta = [], [], []
    per_ep_report = []

    for ei, rec in enumerate(sel):
        ep = rec["episode_index"]
        seg = ddf[(ddf["index"] >= rec["from_index"]) & (ddf["index"] < rec["to_index"])].sort_values("frame_index")
        seg = seg.reset_index(drop=True)
        ep_len = len(seg)
        if ep_len <= chunk + 1:
            print(f"  ep {ep}: too short ({ep_len} <= {chunk + 1}), skip", flush=True)
            continue

        j_act = np.stack(seg[JOINT_ACT].to_numpy())  # [L,7]
        g_act = np.asarray(seg[GRIP_ACT].to_numpy(), dtype=np.float32).reshape(-1)  # [L]
        j_state = np.stack(seg[JOINT_STATE].to_numpy())  # [L,7]
        g_state = np.asarray(seg[GRIP_STATE].to_numpy(), dtype=np.float32).reshape(-1)  # [L]
        task = _task_string(rec, task_map, seg)

        # sparse query frames, leaving room for a full GT chunk after each
        last = ep_len - chunk - 1
        qs = np.unique(np.linspace(0, last, num=min(q_per_ep, last + 1)).astype(int))

        conts = {name: _open_video(root, key) for name, key in CAMS.items()}
        ep_sq = 0.0
        ep_terms = 0
        try:
            for t in qs:
                obs = {"prompt": task}
                for name in CAMS:
                    ts = rec[f"{name}_from_ts"] + float(t) / FPS
                    obs[OBS_KEY[name]] = _frame_at(conts[name], ts)
                obs["observation/joint_position"] = j_state[t][None, :].astype(np.float32)  # [1,7]
                obs["observation/gripper_position"] = np.float32(g_state[t])

                tc = time.time()
                out = svc.infer(obs)
                torch.cuda.synchronize()
                latencies.append(time.time() - tc)

                pred = np.asarray(out["action"], dtype=np.float64)[:chunk, :adim]  # [chunk,8]
                gt = np.concatenate(
                    [j_act[t : t + chunk], g_act[t : t + chunk, None]], axis=-1
                ).astype(np.float64)  # [chunk,8]
                m = min(pred.shape[0], gt.shape[0])
                e = pred[:m] - gt[:m]
                sq_err[:m] += e ** 2
                abs_err[:m] += np.abs(e)
                n_terms += 1
                ep_sq += float((e ** 2).mean())
                ep_terms += 1

                cache_gt.append(gt[:m].astype(np.float32))
                cache_pred.append(pred[:m].astype(np.float32))
                cache_meta.append((int(ep), int(t)))

                # showcase the query with the most GT joint motion, not the idle episode start
                motion = float(gt[:m, :7].std(axis=0).sum())
                if motion > best_motion:
                    best_motion = motion
                    sample_for_plot = (gt[:m].copy(), pred[:m].copy(), {"ep": int(ep), "t": int(t), "task": task})
        finally:
            for c in conts.values():
                c.close()

        ep_rmse = float(np.sqrt(ep_sq / max(ep_terms, 1)))
        per_ep_report.append({"episode": int(ep), "queries": int(ep_terms), "rmse": ep_rmse, "task": task})
        print(f"  [{ei + 1}/{len(sel)}] ep {ep}: len={ep_len} queries={ep_terms} rmse={ep_rmse:.4f}", flush=True)

    if n_terms == 0:
        print("FAIL: no queries evaluated.", file=sys.stderr)
        return 3

    # --- aggregate -------------------------------------------------------------
    per_dim_rmse = np.sqrt(sq_err.sum(axis=0) / (n_terms * chunk))  # [8]
    per_dim_mae = abs_err.sum(axis=0) / (n_terms * chunk)  # [8]
    horizon_rmse = np.sqrt(sq_err.mean(axis=1) / n_terms)  # [chunk]
    overall_rmse = float(np.sqrt(sq_err.sum() / (n_terms * chunk * adim)))
    per_dim_rmse_n = (per_dim_rmse / action_std) if action_std is not None else None

    lat = np.asarray(latencies)
    print("\n=== OPEN-LOOP RESULTS ===", flush=True)
    print(f"episodes evaluated : {len(per_ep_report)}   total queries: {n_terms}   horizon: {chunk}", flush=True)
    print(f"overall RMSE (raw) : {overall_rmse:.4f}", flush=True)
    print("per-dim RMSE / MAE (raw):", flush=True)
    for d, name in enumerate(DIM_LABELS):
        extra = f"  norm={per_dim_rmse_n[d]:.3f}" if per_dim_rmse_n is not None else ""
        print(f"  {name:8s}  rmse={per_dim_rmse[d]:.4f}  mae={per_dim_mae[d]:.4f}{extra}", flush=True)
    print(f"latency/call       : mean={lat.mean():.2f}s  p50={np.median(lat):.2f}s  "
          f"min={lat.min():.2f}s  max={lat.max():.2f}s  (cold={lat[0]:.2f}s)", flush=True)
    print(f"peak VRAM          : {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)

    saved = _plots(out_dir, per_dim_rmse, per_dim_rmse_n, horizon_rmse, sample_for_plot)

    metrics = {
        "checkpoint": ckpt,
        "dataset_root": root,
        "num_episodes": len(per_ep_report),
        "queries_per_episode": q_per_ep,
        "total_queries": int(n_terms),
        "horizon": int(chunk),
        "action_dim": int(adim),
        "num_steps": num_steps,
        "seed": seed,
        "overall_rmse_raw": overall_rmse,
        "per_dim_rmse_raw": {DIM_LABELS[d]: float(per_dim_rmse[d]) for d in range(adim)},
        "per_dim_mae_raw": {DIM_LABELS[d]: float(per_dim_mae[d]) for d in range(adim)},
        "per_dim_rmse_norm": (
            {DIM_LABELS[d]: float(per_dim_rmse_n[d]) for d in range(adim)} if per_dim_rmse_n is not None else None
        ),
        "horizon_rmse": [float(x) for x in horizon_rmse],
        "latency_s": {
            "mean": float(lat.mean()), "p50": float(np.median(lat)),
            "min": float(lat.min()), "max": float(lat.max()), "cold": float(lat[0]),
        },
        "peak_vram_gb": float(torch.cuda.max_memory_allocated() / 1e9),
        "per_episode": per_ep_report,
        "plots": [os.path.basename(p) for p in saved],
    }
    mpath = os.path.join(out_dir, "openloop_metrics.json")
    with open(mpath, "w") as f:
        json.dump(metrics, f, indent=2)

    # cache every per-query GT/pred chunk so plots/analysis can be regenerated without re-inference
    npz = os.path.join(out_dir, "openloop_rollouts.npz")
    np.savez_compressed(
        npz,
        gt=np.stack(cache_gt), pred=np.stack(cache_pred), meta=np.asarray(cache_meta, dtype=np.int64),
        dim_labels=np.asarray(DIM_LABELS),
    )
    print(f"\nwrote {mpath}", flush=True)
    print(f"wrote {npz}", flush=True)
    for p in saved:
        print(f"wrote {p}", flush=True)
    print("PASS: open-loop DROID evaluation complete on ROCm", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
