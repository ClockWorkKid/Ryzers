# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Model-agnostic iTHOR ObjectNav benchmark on Strix Halo (gfx1151).

Runs the seam-selected policy (POLICY_FACTORY) across the iTHOR ObjectNav tasks (4 scenes x 3
targets) x N seeds, in a single process so any model warmup (e.g. AVDC's fp16 + torch.compile) is
paid once and amortized. Records per-task success rate into result_dict.json + summary.json plus one
sample rollout video per task. Mirrors upstream benchmark_thor.py, multi-task and single-process.
"""
import argparse
import json
import os
import time

import numpy as np


def _parse_tasks(spec):
    from sim_ithor.tasks import all_tasks, SCENE2TARGETS

    if not spec or spec == "all":
        return all_tasks()
    out = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        scene, target = item.split(":", 1)
        out.append((scene.strip(), target.strip()))
    # validate
    for scene, target in out:
        assert scene in SCENE2TARGETS and target in SCENE2TARGETS[scene], f"unknown task {scene}:{target}"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=os.environ.get("TASKS") or "all",
                    help="'all' or comma list 'FloorPlan1:Toaster,FloorPlan201:Laptop'")
    ap.add_argument("--n-seeds", type=int, default=int(os.environ.get("N_SEEDS") or "20"))
    ap.add_argument("--resolution", type=int, default=int(os.environ.get("RESOLUTION") or "64"))
    ap.add_argument("--max-eplen", type=int, default=int(os.environ.get("MAX_EPLEN") or "50"))
    ap.add_argument("--render-resolution", type=int,
                    default=(int(os.environ["RENDER_RESOLUTION"]) if os.environ.get("RENDER_RESOLUTION") else None))
    ap.add_argument("--out", default=os.path.join(os.environ.get("OUT_DIR", "/sim_outputs"), "ithor", "benchmark"))
    args = ap.parse_args()

    from sim_ithor.policy import load_policy
    from sim_ithor.rollout import RolloutCfg, run_task

    tasks = _parse_tasks(args.tasks)
    os.makedirs(args.out, exist_ok=True)
    policy = load_policy()
    print(f"benchmark: {len(tasks)} tasks x {args.n_seeds} seeds, res={args.resolution}, "
          f"max_eplen={args.max_eplen}, policy={getattr(policy, 'name', type(policy).__name__)}", flush=True)

    results, t_start = {}, time.time()
    for scene, target in tasks:
        key = f"{scene}:{target}"
        try:
            cfg = RolloutCfg(scene=scene, target=target, output_dir=args.out, n_seeds=args.n_seeds,
                             resolution=args.resolution, max_eplen=args.max_eplen,
                             render_resolution=args.render_resolution, run_tag="benchmark")
            r = run_task(policy, cfg)
            results[key] = {"success_rate": r["success_rate"], "successes": r["successes"],
                            "n_seeds": r["n_seeds"], "eplens": r["eplens"]}
        except Exception as e:  # noqa: BLE001
            print(f"  ERR {key}: {e}", flush=True)
            results[key] = {"success_rate": 0.0, "successes": 0, "n_seeds": args.n_seeds, "error": str(e)}
        print(f"[{key}] success_rate {results[key]['success_rate']:.2f}, "
              f"elapsed {time.time()-t_start:.0f}s", flush=True)
        with open(os.path.join(args.out, "result_dict.json"), "w") as f:
            json.dump(results, f, indent=2)

    rates = [v["success_rate"] for v in results.values()]
    summary = {"mean_success_rate": float(np.mean(rates)) if rates else 0.0,
               "n_tasks": len(results), "n_seeds": args.n_seeds, "resolution": args.resolution,
               "max_eplen": args.max_eplen, "wall_time_s": round(time.time() - t_start, 1),
               "per_task": results}
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nMEAN success rate over {len(results)} tasks: {summary['mean_success_rate']:.3f} "
          f"({summary['wall_time_s']:.0f}s total)")


if __name__ == "__main__":
    main()
