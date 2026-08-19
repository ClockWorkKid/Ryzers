# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Model-agnostic RoboCasa closed-loop evaluation (success-rate benchmark).

Mirrors upstream X-WAM's robocasa_client.py multi-rollout structure (per-rollout seeded
env, run to task horizon or success, save a video, report success rate) but drives the
generic Policy seam instead of a ZMQ broker. This is the closed-loop benchmark entrypoint
a chained policy image points at.

Env: TASK, NUM_EVALS (5), SEED_BASE (0), MAX_STEPS (0=task default), OUT_DIR,
POLICY_FACTORY, VIDEO_RES (512), VIDEO_EVERY (4 executed steps per saved frame).
"""
import json
import os
from datetime import datetime

from sim_robocasa.envutil import env_int, env_str
from sim_robocasa.policy import load_policy
from sim_robocasa.render import banner_frame, compose_view, save_mp4
from sim_robocasa.rollout import run_episode
from sim_robocasa.scene import build_scene


def main():
    task = env_str("TASK", "TurnOnSinkFaucet")
    num_evals = env_int("NUM_EVALS", 5)
    seed_base = env_int("SEED_BASE", 0)
    max_steps = env_int("MAX_STEPS", 0) or None
    video_res = env_int("VIDEO_RES", 512)
    video_every = env_int("VIDEO_EVERY", 4)
    out_dir = env_str("OUT_DIR", "/sim_outputs")
    save_dir = os.path.join(out_dir, "robocasa", task)
    os.makedirs(save_dir, exist_ok=True)

    policy = load_policy()
    print(f"[closedloop] task={task} num_evals={num_evals} policy="
          f"{getattr(policy, 'name', type(policy).__name__)}", flush=True)

    num_success = 0
    for i in range(num_evals):
        seed = seed_base + i
        scene = build_scene(task, seed=seed)
        instr = scene.description
        print(f"[closedloop] rollout {i}/{num_evals} seed={seed}: \"{instr}\"", flush=True)

        frames = []

        def on_frame(view, step, replanning):
            if step % video_every == 0:
                frames.append(banner_frame(view, f"{task}: {instr}", video_res))

        success, model_steps = run_episode(scene, policy, instr, on_frame=on_frame,
                                            max_steps=max_steps)
        num_success += int(success)

        ts = datetime.now().strftime("%H%M%S")
        tag = "success" if success else "failure"
        path = os.path.join(save_dir, f"episode{i}_seed{seed}_{tag}_{ts}.mp4")
        if frames:
            try:
                save_mp4(frames, path, fps=10)
                print(f"[closedloop]   -> {tag} ({model_steps} model calls); saved {path}", flush=True)
            except Exception as e:  # noqa: BLE001
                print("[closedloop]   video save failed:", e, flush=True)
        else:
            print(f"[closedloop]   -> {tag} ({model_steps} model calls); no frames", flush=True)
        scene.close()

    rate = num_success / max(1, num_evals)
    print(f"[closedloop] RESULT {task}: {num_success}/{num_evals} => {100.0 * rate:.1f}%", flush=True)
    with open(os.path.join(save_dir, "_result.json"), "w") as f:
        json.dump({"task": task, "num_success": num_success, "num_evals": num_evals,
                   "success_rate": rate}, f, indent=2)


if __name__ == "__main__":
    main()
