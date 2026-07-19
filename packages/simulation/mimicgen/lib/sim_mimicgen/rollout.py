# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Model-agnostic closed-loop rollout over the MimicGen simulator.

Thin wrapper around the upstream `vera.env_runner.MimicgenRunner` (reused unchanged, rule 2.1):
it resets to each demo's initial state (from the task hdf5), warms up the context window, drives
the given seam `Policy` per env step, tracks success / max reward, and writes per-view videos.
Task identity comes entirely from the robosuite/mimicgen hdf5 (env config + demo initial states),
so `--dataset <task>.hdf5` fully determines the task.

The runner's view/context/render config is taken from the policy's advertised hints
(`view_keys`, `context_frames`, `render_size`) when present — a remote model self-describes via
server metadata — otherwise from the explicit args (RandomPolicy sanity path).
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

logger = logging.getLogger("sim_mimicgen.rollout")

# Sensible defaults for the RandomPolicy sanity path (2-view MimicGen, matches VERA's setup).
DEFAULT_VIEW_KEYS = ["agentview_image", "robot0_eye_in_hand_image"]
DEFAULT_CONTEXT_FRAMES = 9


@dataclass
class RolloutCfg:
    dataset: str                      # robosuite/mimicgen hdf5, e.g. .../core/stack_d0.hdf5
    output_dir: str = "/sim_outputs/mimicgen"
    num_demos: int = 1
    rollout_horizon: int = 400
    render_size: int = 128
    view_keys: Optional[List[str]] = None
    context_frames: Optional[int] = None
    action_scale: float = 1.0
    save_videos: bool = True
    run_tag: str = "sim_mimicgen"


def run_rollout(policy, cfg: RolloutCfg) -> dict:
    """Run a closed-loop MimicGen eval driving `policy` (a sim_mimicgen.Policy / BasePolicy)."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    from vera.env_runner.mimicgen_runner import MimicgenRunner, MimicgenRunnerCfg

    # Prefer the policy's self-advertised hints (a remote model reads these from server metadata).
    view_keys = list(cfg.view_keys or getattr(policy, "view_keys", None) or DEFAULT_VIEW_KEYS)
    context_frames = int(
        cfg.context_frames or getattr(policy, "context_frames", None) or DEFAULT_CONTEXT_FRAMES
    )
    render_size = int(cfg.render_size or getattr(policy, "render_size", None) or 128)

    logger.info("mimicgen rollout: dataset=%s demos=%d H=%d views=%s ctx=%d render=%d",
                os.path.basename(cfg.dataset), cfg.num_demos, cfg.rollout_horizon,
                view_keys, context_frames, render_size)

    runner_cfg = MimicgenRunnerCfg(
        env_name="mimicgen", dataset_path=cfg.dataset, render_size=render_size,
        render_obs_key=view_keys, num_demos_to_run=int(cfg.num_demos),
        max_episode_steps=int(cfg.rollout_horizon), n_repeat=1, action_scale=float(cfg.action_scale),
        save_videos=bool(cfg.save_videos), save_trajectory=False, save_rrd=False,
        output_dir=cfg.output_dir, use_stored_model=False,
        demo_warmup_steps=max(context_frames - 1, 0), log_step_debug=False,
    )
    runner = MimicgenRunner(runner_cfg, device="cpu")
    runner.setup_env()

    result = runner.run(policy, run_tag=cfg.run_tag)

    succ = np.asarray(result.get("env_successes", []), dtype=bool)
    relaxed = np.asarray(result.get("relaxed_successes", succ), dtype=bool)
    max_r = np.asarray(result.get("max_rewards", []), dtype=float)
    n = len(succ)
    logger.info("=" * 60)
    logger.info("EVAL DONE: %s", os.path.basename(cfg.dataset))
    logger.info("  success rate: %d/%d = %.1f%%", int(succ.sum()), n, 100.0 * succ.mean() if n else 0.0)
    if relaxed.size:
        logger.info("  relaxed success rate: %d/%d = %.1f%%", int(relaxed.sum()), n, 100.0 * relaxed.mean())
    if max_r.size:
        logger.info("  max reward mean: %.3f", float(max_r.mean()))
    logger.info("  videos/results: %s", result.get("save_dir"))
    logger.info("=" * 60)
    return result
