# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Headless sanity rollout for the RoboCasa simulator base image.

Loads a policy (default: built-in RandomPolicy), rolls it through one kitchen scene for a
bounded number of steps, and saves an MP4 of the composed 3-view. Proves the ROCm/EGL
render + MuJoCo step + video encode path work end-to-end with no model.

Env: TASK, SEED, STEPS, OUT_DIR, POLICY_FACTORY (module:function).
"""
import os
from datetime import datetime

from sim_robocasa.envutil import env_int, env_str
from sim_robocasa.policy import load_policy
from sim_robocasa.render import banner_frame, compose_view, save_mp4
from sim_robocasa.rollout import run_episode
from sim_robocasa.scene import build_scene


def main():
    task = env_str("TASK", "TurnOnSinkFaucet")
    seed = env_int("SEED", 0)
    steps = env_int("STEPS", 80)
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
