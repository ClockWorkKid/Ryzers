# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""AVDC closed-loop iTHOR ObjectNav rollout on Strix Halo (gfx1151), single (scene, target).

Drives one iTHOR ObjectNav episode with AVDC's video->flow->nav policy over the sim_ithor harness
(headless ai2thor CloudRendering / Vulkan). Saves the executed rollout video + metrics, and a
rule-2.b two-column debug GIF (left: the current sim observation the plan was imagined from; right:
AVDC's generated future video plan). Single-config validation wrapper around the upstream
benchmark_thor.py loop.
"""
import argparse
import json
import os

import numpy as np
import imageio.v2 as imageio


def _two_column(left_frames, right_frames, path, fps=5):
    """Save a side-by-side GIF (rule 2.b: reference/sim left, generated plan right)."""
    def to_hwc(f):
        f = np.asarray(f)
        if f.ndim == 3 and f.shape[0] in (1, 3) and f.shape[2] not in (1, 3):
            f = np.transpose(f, (1, 2, 0))  # CHW -> HWC (upstream plan frames are channel-first)
        return f.astype(np.uint8)

    left = [to_hwc(f) for f in left_frames]
    right = [to_hwc(f) for f in right_frames]
    n = max(len(left), len(right))
    if not left or not right:
        return
    H = max(left[0].shape[0], right[0].shape[0])

    def resize_to_h(f, H):
        from PIL import Image
        w = int(round(f.shape[1] * H / f.shape[0]))
        return np.asarray(Image.fromarray(f).resize((w, H)))

    cols = []
    for i in range(n):
        lf = resize_to_h(left[min(i, len(left) - 1)], H)
        rf = resize_to_h(right[min(i, len(right) - 1)], H)
        pad = np.zeros((H, 8, 3), dtype=np.uint8)
        cols.append(np.concatenate([lf, pad, rf], axis=1).astype(np.uint8))
    imageio.mimsave(path, cols, duration=1000.0 / max(fps, 1), loop=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=os.environ.get("SCENE") or "FloorPlan1")
    ap.add_argument("--target", default=os.environ.get("TARGET") or "Toaster")
    ap.add_argument("--seed", type=int, default=int(os.environ.get("SEED") or "0"))
    ap.add_argument("--resolution", type=int, default=int(os.environ.get("RESOLUTION") or "64"))
    ap.add_argument("--max-eplen", type=int, default=int(os.environ.get("MAX_EPLEN") or "50"))
    ap.add_argument("--render-resolution", type=int,
                    default=(int(os.environ["RENDER_RESOLUTION"]) if os.environ.get("RENDER_RESOLUTION") else 256))
    ap.add_argument("--out", default=os.path.join(os.environ.get("OUT_DIR", "/outputs"), "ithor"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    from sim_ithor.env import ThorEnv
    from avdc_ithor_policy import build_policy

    policy = build_policy()
    policy.reset(args.target)

    env = ThorEnv(args.scene, args.target, seed=args.seed,
                  resolution=(args.resolution, args.resolution), max_eplen=args.max_eplen)
    render_env = ThorEnv(args.scene, args.target, seed=args.seed,
                         resolution=(args.render_resolution, args.render_resolution),
                         max_eplen=args.max_eplen)
    try:
        frame, depth = env.reset()
        render_env.seed(args.seed)
        rframe, _ = render_env.reset()
        frames = [rframe]

        # Capture AVDC's first generated video plan from the reset observation for the 2.b viz.
        first_plan = None
        try:
            from flowdiffusion.inference_utils import pred_video_thor
            first_plan = pred_video_thor(policy.video_model, frame, args.target)
        except Exception as e:  # noqa: BLE001
            print(f"[demo_ithor] plan-viz capture skipped ({e})")

        success, done, n_plans = False, False, 0
        while not done:
            actions = policy.plan((frame, depth))
            n_plans += 1
            if not actions:
                break
            for action in actions:
                (frame, depth), success, done = env.step(action)
                (rframe, _), _, _ = render_env.step(action)
                frames.append(rframe)
                if done:
                    break
        eplen = int(env.eplen)
        print(f"rollout: scene={args.scene} target={args.target} seed={args.seed} "
              f"eplen={eplen} success={success} plans={n_plans}")

        tag = f"{args.scene}_{args.target}_{args.seed}"
        vid = os.path.join(args.out, f"{tag}.mp4")
        try:
            imageio.mimsave(vid, [np.asarray(f).astype(np.uint8) for f in frames], fps=5)
        except Exception:
            vid = vid[:-4] + ".gif"
            imageio.mimsave(vid, [np.asarray(f).astype(np.uint8) for f in frames], duration=200, loop=0)
        print(f"saved rollout -> {vid}")

        if first_plan is not None:
            init_left = [np.asarray(frames[0])] * len(first_plan)
            twocol = os.path.join(args.out, f"{tag}_plan_vs_sim.gif")
            try:
                _two_column(init_left, first_plan, twocol)
                print(f"saved plan-vs-sim (rule 2.b) -> {twocol}")
            except Exception as e:  # noqa: BLE001
                print(f"[demo_ithor] two-column viz failed ({e})")

        with open(os.path.join(args.out, f"{tag}_metrics.json"), "w") as f:
            json.dump({"scene": args.scene, "target": args.target, "seed": args.seed,
                       "eplen": eplen, "success": bool(success), "n_plans": n_plans,
                       "max_eplen": args.max_eplen}, f, indent=2)
    finally:
        env.close()
        render_env.close()


if __name__ == "__main__":
    main()
