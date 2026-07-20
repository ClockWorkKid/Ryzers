# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""CLI: run a closed-loop iTHOR ObjectNav rollout with the seam-selected policy.

  python -m sim_ithor.sanity --scene FloorPlan1 --target Toaster

Default policy is the built-in ScriptedPolicy (no model) - a headless sign-of-life that starts the
ai2thor CloudRendering (Vulkan) controller, resets to a random reachable pose, steps discrete
moves, renders offscreen and writes a video. Set POLICY_FACTORY=module:function (e.g. a model
layer's adapter) to drive a real model.
"""
from __future__ import annotations

import argparse
import json
import logging
import os

from sim_ithor.policy import load_policy
from sim_ithor.rollout import RolloutCfg, run_task

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")


def main() -> int:
    ap = argparse.ArgumentParser(description="iTHOR ObjectNav harness rollout (seam policy)")
    ap.add_argument("--scene", default=os.environ.get("SCENE") or "FloorPlan1")
    ap.add_argument("--target", default=os.environ.get("TARGET") or "Toaster")
    ap.add_argument("--output-dir", default=os.path.join(os.environ.get("OUT_DIR", "/sim_outputs"), "ithor"))
    ap.add_argument("--n-seeds", type=int, default=int(os.environ.get("N_SEEDS") or "1"))
    ap.add_argument("--resolution", type=int, default=int(os.environ.get("RESOLUTION") or "64"))
    ap.add_argument("--max-eplen", type=int, default=int(os.environ.get("MAX_EPLEN") or "50"))
    ap.add_argument("--render-resolution", type=int,
                    default=(int(os.environ["RENDER_RESOLUTION"]) if os.environ.get("RENDER_RESOLUTION") else None))
    ap.add_argument("--run-tag", default="sanity")
    args = ap.parse_args()

    policy = load_policy()
    logging.getLogger("sim_ithor.sanity").info("policy: %s", getattr(policy, "name", type(policy).__name__))
    cfg = RolloutCfg(
        scene=args.scene, target=args.target, output_dir=args.output_dir, n_seeds=args.n_seeds,
        resolution=args.resolution, max_eplen=args.max_eplen,
        render_resolution=args.render_resolution, run_tag=args.run_tag,
    )
    result = run_task(policy, cfg)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, f"{args.scene}_{args.target}_metrics.json"), "w") as f:
        json.dump(result, f, indent=2)
    print("RESULT:", json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
