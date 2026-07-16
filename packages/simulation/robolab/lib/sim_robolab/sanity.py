# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Headless sanity rollout for the RoboLab-AMD simulator base image.

Loads a policy (default: built-in joint-space RandomPolicy), rolls it through one RoboLab
task for a bounded number of steps, and saves an MP4 of the composed 3-view. Proves the
ROCm/EGL render + MuJoCo step (JOINT_POSITION control) + video encode path work end-to-end
with no model (gate G1).

Env: TASK, SEED, STEPS, OUT_DIR, POLICY_FACTORY (module:function).
"""
import os
from datetime import datetime

from sim_robolab.envutil import env_int, env_str
from sim_robolab.policy import load_policy
from sim_robolab.render import banner_frame, compose_view, save_mp4
from sim_robolab.rollout import run_episode
from sim_robolab.scene import build_scene


def main():
    task = env_str("TASK", "BananaInBowl")
    seed = env_int("SEED", 0)
    steps = env_int("STEPS", 120)
    out_dir = env_str("OUT_DIR", "/sim_outputs")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[sanity] building scene {task} (seed={seed}) ...", flush=True)
    scene = build_scene(task, seed=seed)
    print(f"[sanity] scene ready: \"{scene.description}\"", flush=True)

    policy = load_policy()
    print(f"[sanity] policy: {getattr(policy, 'name', type(policy).__name__)}", flush=True)

    frames = []

    def on_frame(view, step, replanning):
        frames.append(banner_frame(compose_view(view), f"{policy.name}: {scene.description}", 720))

    success, model_steps = run_episode(
        scene, policy, scene.description, on_frame=on_frame, max_steps=steps
    )

    ts = datetime.now().strftime("%H%M%S")
    path = os.path.join(out_dir, f"sanity_{task}_{ts}.mp4")
    if frames:
        save_mp4(frames, path, fps=20)
        print(f"[sanity] OK: {len(frames)} frames, {model_steps} model calls, success={success}", flush=True)
        print(f"[sanity] saved {path}", flush=True)
    else:
        raise RuntimeError("no frames rendered")

    scene.close()


if __name__ == "__main__":
    main()
