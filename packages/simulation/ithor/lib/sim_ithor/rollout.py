# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Model-agnostic closed-loop ObjectNav rollout over the iTHOR simulator.

Drives a seam `Policy` over `sim_ithor.env.ThorEnv`, mirroring the upstream
AVDC_experiments benchmark_thor.py `eval()` loop (reset to a random reachable pose, plan-then-
execute discrete moves, stop on target-visible or the step cap), tracking success / episode length
and writing a sample rollout video. Task identity is `(scene, target)`; success = the target object
became `visible` before `max_eplen` steps.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

logger = logging.getLogger("sim_ithor.rollout")


@dataclass
class RolloutCfg:
    scene: str
    target: str
    output_dir: str = "/sim_outputs/ithor"
    n_seeds: int = 1
    resolution: int = 64
    max_eplen: int = 50
    rotate_step_degrees: int = 45
    visibility_distance: float = 1.5
    render_resolution: Optional[int] = None   # None => record obs-resolution frames
    save_video: bool = True
    fps: int = 5
    run_tag: str = "sim_ithor"


def _save_video(frames, path, fps):
    import imageio.v2 as imageio

    frames = [np.asarray(f).astype(np.uint8) for f in frames]
    if path.endswith(".gif"):
        imageio.mimsave(path, frames, duration=1000.0 / max(fps, 1), loop=0)
    else:
        imageio.mimsave(path, frames, fps=fps)


def run_episode(env, policy, target, render_env=None) -> dict:
    """One ObjectNav episode: plan-then-execute until success/Done/step-cap. Returns metrics+frames."""
    frame, depth = env.reset()
    if render_env is not None:
        render_env.seed(env.rng.randint(int(1e6)))
        rframe, _ = render_env.reset()
        frames = [rframe]
    else:
        frames = [frame]
    policy.reset(target)

    success, done, n_plans = False, False, 0
    while not done:
        actions = policy.plan((frame, depth))
        n_plans += 1
        if not actions:
            break
        for action in actions:
            (frame, depth), success, done = env.step(action)
            if render_env is not None:
                (rframe, _), _, _ = render_env.step(action)
                frames.append(rframe)
            else:
                frames.append(frame)
            if done:
                break
    return {"success": bool(success), "eplen": int(env.eplen), "n_plans": int(n_plans),
            "frames": frames}


def run_task(policy, cfg: RolloutCfg) -> dict:
    """Run `n_seeds` ObjectNav episodes for one (scene, target); aggregate success rate."""
    from sim_ithor.env import ThorEnv

    os.makedirs(cfg.output_dir, exist_ok=True)
    logger.info("ithor rollout: scene=%s target=%s seeds=%d res=%d max_eplen=%d",
                cfg.scene, cfg.target, cfg.n_seeds, cfg.resolution, cfg.max_eplen)

    env = ThorEnv(cfg.scene, cfg.target, seed=0, resolution=(cfg.resolution, cfg.resolution),
                  max_eplen=cfg.max_eplen, rotate_step_degrees=cfg.rotate_step_degrees,
                  visibility_distance=cfg.visibility_distance)
    render_env = None
    if cfg.render_resolution and cfg.render_resolution != cfg.resolution:
        render_env = ThorEnv(cfg.scene, cfg.target, seed=0,
                             resolution=(cfg.render_resolution, cfg.render_resolution),
                             max_eplen=cfg.max_eplen, rotate_step_degrees=cfg.rotate_step_degrees,
                             visibility_distance=cfg.visibility_distance)

    successes, eplens = 0, []
    try:
        for seed in range(cfg.n_seeds):
            env.seed(seed)
            res = run_episode(env, policy, cfg.target, render_env=render_env)
            successes += int(res["success"])
            eplens.append(res["eplen"])
            logger.info("  seed=%d success=%s eplen=%d plans=%d",
                        seed, res["success"], res["eplen"], res["n_plans"])
            if seed == 0 and cfg.save_video:
                vid = os.path.join(cfg.output_dir, f"{cfg.scene}_{cfg.target}.mp4")
                try:
                    _save_video(res["frames"], vid, cfg.fps)
                    logger.info("  saved sample rollout -> %s", vid)
                except Exception as e:  # noqa: BLE001
                    logger.warning("  video save failed (%s); trying gif", e)
                    _save_video(res["frames"], vid[:-4] + ".gif", cfg.fps)
    finally:
        env.close()
        if render_env is not None:
            render_env.close()

    rate = successes / cfg.n_seeds if cfg.n_seeds else 0.0
    logger.info("EVAL %s/%s: success %d/%d = %.1f%%",
                cfg.scene, cfg.target, successes, cfg.n_seeds, 100.0 * rate)
    return {"scene": cfg.scene, "target": cfg.target, "success_rate": rate,
            "successes": successes, "n_seeds": cfg.n_seeds, "eplens": eplens}
