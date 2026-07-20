# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""AVDC closed-loop rollout in the Meta-World MuJoCo simulator on Strix Halo (gfx1151).

Drives one Meta-World V2 goal-observable task with AVDC's closed-loop policy (MyPolicy_CL): at each
replan the video-diffusion model imagines a future clip from the current frame + task, UniMatch
computes dense optical flow between generated frames, and the flow + depth/segmentation are solved
into end-effector actions (rigid transform). Renders headless via EGL on the iGPU. Saves the
executed rollout GIF + metrics (episode length, success, return). This is a single-config wrapper
around the upstream experiment/benchmark_mw.py loop for validation; the full benchmark sweeps 11
tasks x 25 seeds x 3 cameras.
"""
import argparse, json, os

import numpy as np
import imageio.v2 as imageio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-name", default=os.environ.get("ENV_NAME") or "door-open-v2-goal-observable")
    ap.add_argument("--seed", type=int, default=int(os.environ.get("SEED") or "0"))
    ap.add_argument("--camera", default=os.environ.get("CAMERA") or "corner")
    ap.add_argument("--ckpt-dir", default=os.environ.get("CKPT_DIR", "/models/metaworld"))
    ap.add_argument("--milestone", type=int, default=int(os.environ.get("MILESTONE") or "24"))
    ap.add_argument("--sample-steps", type=int, default=int(os.environ.get("SAMPLE_STEPS") or "20"))
    ap.add_argument("--max-replans", type=int, default=int(os.environ.get("MAX_REPLANS") or "5"))
    ap.add_argument("--out", default=os.path.join(os.environ.get("OUT_DIR", "/outputs"), "metaworld"))
    args = ap.parse_args()

    import torch, random
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    from mypolicy import MyPolicy_CL
    from metaworld_exp.utils import collect_video
    from myutils import get_flow_model
    from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE as env_dict
    from flowdiffusion.inference_utils import get_video_model

    print(f"video model      : {args.ckpt_dir}/model-{args.milestone}.pt (ddim steps={args.sample_steps})")
    video_model = get_video_model(ckpts_dir=args.ckpt_dir, milestone=args.milestone, timestep=args.sample_steps)
    import avdc_optim; avdc_optim.apply(video_model)   # env-gated fp16 / torch.compile (phase 4)
    print("flow model       : UniMatch GMFlow (pretrained)")
    flow_model = get_flow_model()

    env = env_dict[args.env_name](seed=args.seed)
    obs = env.reset()
    print(f"env ready        : {args.env_name} seed={args.seed} camera={args.camera} max_replans={args.max_replans}")
    policy = MyPolicy_CL(env, args.env_name, args.camera, video_model, flow_model, max_replans=args.max_replans)

    images, _depths, ep_return = collect_video(obs, env, policy, camera_name=args.camera, resolution=(320, 240))
    images = np.asarray(images).astype(np.uint8)
    eplen = int(len(images))
    success = bool(eplen <= 500)   # upstream benchmark_mw.py criterion: solved before the 500-step cap
    used_replans = int(args.max_replans - getattr(policy, "replans", args.max_replans))
    print(f"rollout          : eplen={eplen} success={success} return={ep_return:.2f} used_replans={used_replans}")

    gif = os.path.join(args.out, f"{args.env_name}_{args.camera}_{args.seed}.gif")
    imageio.mimsave(gif, list(images), duration=1000.0 / 20, loop=0)
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump({"env": args.env_name, "seed": args.seed, "camera": args.camera,
                   "sample_steps": args.sample_steps, "max_replans": args.max_replans,
                   "eplen": eplen, "success": success, "episode_return": float(ep_return),
                   "used_replans": used_replans}, f, indent=2)
    print(f"saved -> {gif}")


if __name__ == "__main__":
    main()
