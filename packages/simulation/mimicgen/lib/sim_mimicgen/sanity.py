# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""CLI: run a closed-loop MimicGen rollout with the seam-selected policy.

  python -m sim_mimicgen.sanity --dataset /models/mimicgen_datasets/core/stack_d0.hdf5

Default policy is the built-in RandomPolicy (no model, no server) — a headless sign-of-life
that steps robosuite/MuJoCo, resets to demo states, renders offscreen, and writes videos.
Set POLICY_FACTORY=module:function (e.g. a model layer's adapter, or
sim_mimicgen.remote_policy:build_policy pointed at a running policy server) to drive a real model.
"""
from __future__ import annotations

import argparse
import logging

from sim_mimicgen.policy import load_policy
from sim_mimicgen.rollout import RolloutCfg, run_rollout

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")


def main() -> int:
    ap = argparse.ArgumentParser(description="MimicGen simulator harness rollout (seam policy)")
    ap.add_argument("--dataset", required=True, help="robosuite/mimicgen hdf5 (e.g. stack_d0.hdf5)")
    ap.add_argument("--output-dir", default="/sim_outputs/mimicgen")
    ap.add_argument("--num-demos", type=int, default=1)
    ap.add_argument("--rollout-horizon", type=int, default=200)
    ap.add_argument("--render-size", type=int, default=128)
    ap.add_argument("--views", default=None, help="comma-separated image obs keys (default: policy hint / 2-view)")
    ap.add_argument("--context-frames", type=int, default=None)
    ap.add_argument("--run-tag", default="sim_mimicgen")
    args = ap.parse_args()

    policy = load_policy()
    cfg = RolloutCfg(
        dataset=args.dataset, output_dir=args.output_dir, num_demos=args.num_demos,
        rollout_horizon=args.rollout_horizon, render_size=args.render_size,
        view_keys=(args.views.split(",") if args.views else None),
        context_frames=args.context_frames, run_tag=args.run_tag,
    )
    run_rollout(policy, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
