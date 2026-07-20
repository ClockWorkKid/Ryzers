# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""AVDC closed-loop Meta-World benchmark on Strix Halo (gfx1151).

Runs the AVDC closed-loop policy across multiple Meta-World V2 goal-observable tasks x seeds (one
camera) in a single process, so the fp16 + torch.compile setup (avdc_optim) is paid once and
amortized across all rollouts. Records per-task success rate (episode solved within the 500-step
cap, matching upstream benchmark_mw.py) into result_dict.json plus one sample rollout GIF per task.
Mirrors experiment/benchmark_mw.py but multi-task and single-process for efficient benchmarking.
"""
import argparse, json, os, time

import numpy as np
import imageio.v2 as imageio

# The 11 tasks in upstream benchmark_mw.sh (order preserved).
DEFAULT_TASKS = [
    "door-open", "door-close", "basketball", "shelf-place", "button-press",
    "button-press-topdown", "faucet-close", "faucet-open", "handle-press", "hammer", "assembly",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=os.environ.get("TASKS") or "all")
    ap.add_argument("--n-seeds", type=int, default=int(os.environ.get("N_SEEDS") or "5"))
    ap.add_argument("--camera", default=os.environ.get("CAMERA") or "corner")
    ap.add_argument("--ckpt-dir", default=os.environ.get("CKPT_DIR", "/models/metaworld"))
    ap.add_argument("--milestone", type=int, default=int(os.environ.get("MILESTONE") or "24"))
    ap.add_argument("--sample-steps", type=int, default=int(os.environ.get("SAMPLE_STEPS") or "10"))
    ap.add_argument("--max-replans", type=int, default=int(os.environ.get("MAX_REPLANS") or "5"))
    ap.add_argument("--out", default=os.path.join(os.environ.get("OUT_DIR", "/outputs"), "benchmark"))
    args = ap.parse_args()

    import torch, random
    from mypolicy import MyPolicy_CL
    from metaworld_exp.utils import collect_video
    from myutils import get_flow_model
    from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE as env_dict
    from flowdiffusion.inference_utils import get_video_model
    import avdc_optim

    tasks = DEFAULT_TASKS if args.tasks == "all" else [t.strip() for t in args.tasks.split(",") if t.strip()]
    os.makedirs(args.out, exist_ok=True)
    print(f"benchmark: {len(tasks)} tasks x {args.n_seeds} seeds, camera={args.camera}, "
          f"ddim={args.sample_steps}, max_replans={args.max_replans}")

    video_model = get_video_model(ckpts_dir=args.ckpt_dir, milestone=args.milestone, timestep=args.sample_steps)
    avdc_optim.apply(video_model)
    flow_model = get_flow_model()

    results, t_start = {}, time.time()
    for task in tasks:
        env_name = f"{task}-v2-goal-observable"
        succ, eplens = 0, []
        for seed in range(args.n_seeds):
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            try:
                env = env_dict[env_name](seed=seed)
                obs = env.reset()
                policy = MyPolicy_CL(env, env_name, args.camera, video_model, flow_model,
                                     max_replans=args.max_replans)
                images, _d, _r = collect_video(obs, env, policy, camera_name=args.camera, resolution=(320, 240))
                eplen = int(len(images))
                if eplen <= 500:
                    succ += 1
                eplens.append(eplen)
                if seed == 0:
                    imageio.mimsave(os.path.join(args.out, f"{env_name}.gif"),
                                    list(np.asarray(images).astype(np.uint8)), duration=1000.0 / 20, loop=0)
            except Exception as e:
                print(f"  ERR {env_name} seed={seed}: {e}")
                eplens.append(-1)
        results[env_name] = {"success_rate": succ / args.n_seeds, "successes": succ,
                             "n_seeds": args.n_seeds, "eplens": eplens}
        print(f"[{env_name}] success {succ}/{args.n_seeds} "
              f"(rate {results[env_name]['success_rate']:.2f}), elapsed {time.time()-t_start:.0f}s", flush=True)
        # write incrementally so a long run is inspectable / resumable
        with open(os.path.join(args.out, "result_dict.json"), "w") as f:
            json.dump(results, f, indent=2)

    rates = [v["success_rate"] for v in results.values()]
    summary = {"mean_success_rate": float(np.mean(rates)) if rates else 0.0,
               "n_tasks": len(results), "n_seeds": args.n_seeds, "camera": args.camera,
               "sample_steps": args.sample_steps, "max_replans": args.max_replans,
               "wall_time_s": round(time.time() - t_start, 1), "per_task": results}
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nMEAN success rate over {len(results)} tasks: {summary['mean_success_rate']:.3f} "
          f"({summary['wall_time_s']:.0f}s total)")


if __name__ == "__main__":
    main()
