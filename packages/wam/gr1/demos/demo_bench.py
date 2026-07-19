# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""GR-1 inference benchmark on ROCm/gfx1151 (phase 4): measures per-step latency AND action
quality in one pass, so every optimization config is scored on both axes (a speedup that hurts
the arm MAE fails the quality gate).

It drives the EXACT closed-loop code path (upstream GR1CalvinEvaluation.step, with the phase-4
optimizations from gr1_optim.py applied) but feeds a real CALVIN episode offline (no simulator),
timing each step with CUDA sync and comparing the predicted 7-DoF action to the ground-truth
rel_actions. Config is selected via env (GR1_AMP / GR1_SDPA / GR1_COMPILE); see gr1_optim.py.
"""
import argparse, json, os, time, statistics
import numpy as np
import torch

from evaluation.calvin_evaluation import GR1CalvinEvaluation
from gr1_optim import apply_optimizations


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mae-ckpt", default=os.environ.get("MAE_CKPT", "/models/mae_pretrain_vit_base.pth"))
    ap.add_argument("--policy-ckpt", default=os.environ.get("POLICY_CKPT", "/models/snapshot_ABCD.pt"))
    ap.add_argument("--configs", default=os.environ.get("CONFIGS", "/repos/gr1/logs/configs.json"))
    ap.add_argument("--data-dir", default=os.environ.get("DATASET_DIR", "/data/calvin_debug_dataset"))
    ap.add_argument("--split", default="validation")
    ap.add_argument("--window-idx", type=int, default=0)
    ap.add_argument("--steps", type=int, default=int(os.environ.get("BENCH_STEPS") or "40"))
    ap.add_argument("--warmup", type=int, default=int(os.environ.get("BENCH_WARMUP") or "8"))
    ap.add_argument("--tag", default=os.environ.get("GR1_TAG") or "run")
    ap.add_argument("--out", default=os.path.join(os.environ.get("OUT_DIR", "/outputs"), "bench"))
    ap.add_argument("--seed", type=int, default=int(os.environ.get("SEED") or "0"))
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda", 0)

    with open(args.configs) as f:
        variant = json.load(f)

    model = GR1CalvinEvaluation(args.mae_ckpt, args.policy_ckpt, variant, device)
    label, cfg = apply_optimizations(model)

    split_dir = os.path.join(args.data_dir, args.split)
    ann = np.load(os.path.join(split_dir, "lang_annotations", "auto_lang_ann.npy"),
                  allow_pickle=True).item()
    langs, indx = ann["language"]["ann"], ann["info"]["indx"]
    wi = args.window_idx % len(langs)
    start, end = int(indx[wi][0]), int(indx[wi][1])
    lang = str(langs[wi])
    n_frames = min(args.steps, end - start + 1)
    frames = list(range(start, start + n_frames))
    print(f"[bench:{label}] window {wi} '{lang}'  {n_frames} steps (warmup {args.warmup})")

    model.reset()
    lat_ms, pred_arm, gt_arm, pred_grip, gt_grip = [], [], [], [], []
    for k, t in enumerate(frames):
        ep = np.load(os.path.join(split_dir, f"episode_{t:07d}.npz"))
        obs = {"rgb_obs": {"rgb_static": ep["rgb_static"], "rgb_gripper": ep["rgb_gripper"]},
               "robot_obs": ep["robot_obs"].astype(np.float32)}
        rel = ep["rel_actions"].astype(np.float32)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        action = model.step(obs, lang)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000.0
        if k >= args.warmup:
            lat_ms.append(dt)
            a = action.numpy()
            pred_arm.append(a[:6]); pred_grip.append(1.0 if a[6] > 0 else -1.0)
            gt_arm.append(rel[:6]); gt_grip.append(float(rel[6]))

    pred_arm, gt_arm = np.array(pred_arm), np.array(gt_arm)
    arm_mae = float(np.abs(pred_arm - gt_arm).mean())
    grip_match = float((np.sign(pred_grip) == np.sign(gt_grip)).mean())
    lat_ms.sort()
    mean_ms = statistics.mean(lat_ms); med_ms = statistics.median(lat_ms)
    p90_ms = lat_ms[int(0.9 * (len(lat_ms) - 1))]
    fps = 1000.0 / mean_ms

    res = {"tag": args.tag, "label": label, "config": cfg,
           "timed_steps": len(lat_ms), "warmup": args.warmup,
           "mean_ms": round(mean_ms, 2), "median_ms": round(med_ms, 2), "p90_ms": round(p90_ms, 2),
           "fps": round(fps, 2), "arm_mae": round(arm_mae, 4), "gripper_match": round(grip_match, 4),
           "torch": torch.__version__, "hip": torch.version.hip}
    print(json.dumps(res, indent=2))
    with open(os.path.join(args.out, f"bench_{args.tag}.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"saved -> {os.path.join(args.out, f'bench_{args.tag}.json')}")


if __name__ == "__main__":
    main()
