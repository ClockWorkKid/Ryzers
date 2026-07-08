# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Headless render/sim sanity: run a short RandomPolicy rollout on one SimplerEnv task and
write an MP4. This is the make-or-break validation of SAPIEN offscreen Vulkan rendering +
ManiSkill2 CPU physics on gfx1151 (no model weights involved).

  TASK=google_robot_pick_coke_can STEPS=40 python -m sim_simplerenv.sanity
"""
import os

import imageio
import numpy as np

from .envutil import env_int, env_str
from . import simplerenv_env as se
from .policy import load_policy


def main() -> int:
    task = env_str("TASK", "google_robot_pick_coke_can")
    steps = env_int("STEPS", 40)
    seed = env_int("SEED", 0)
    out_dir = os.path.join(env_str("OUT_DIR", "/sim_outputs"), "simplerenv")
    os.makedirs(out_dir, exist_ok=True)

    print(f"task={task}  steps={steps}  seed={seed}  policy_setup={se.policy_setup_for(task)}", flush=True)
    env, obs, instruction = se.build_env(task, seed=seed)
    print(f"instruction: {instruction!r}", flush=True)
    policy = load_policy()
    policy.reset(instruction)

    frames, ok = [], False
    for t in range(steps):
        frames.append(se.get_image(env, obs))
        chunk = np.asarray(policy.predict_action_chunk(obs, instruction))
        for a in chunk[: policy.replan_steps]:
            obs, _reward, terminated, truncated, info = env.step(np.asarray(a, dtype=np.float32))
            if se.is_success(info, terminated):
                ok = True
            if terminated or truncated:
                break
        if terminated or truncated:
            break

    path = os.path.join(out_dir, f"sanity_{task}_{'ok' if ok else 'run'}.mp4")
    imageio.mimwrite(path, [np.asarray(f) for f in frames], fps=10)
    env.close()
    print(f"wrote {path}  ({len(frames)} frames, success={ok})", flush=True)
    print("PASS: SimplerEnv headless render/sim sanity OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
