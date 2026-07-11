# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Model-agnostic SimplerEnv episode loop (chunk-replay).

Mirrors the closed-loop rollout against the sim_simplerenv.Policy seam: get an observation,
ask the policy for a [T, 7] delta-pose + gripper chunk, execute the first ``replan_steps``
rows via env.step, then replan. Each executed step yields the live third-person frame to the
on_frame callback. No torch or model code here.
"""
import numpy as np

from . import simplerenv_env as se


def run_episode(env, obs, policy, instruction, on_frame=None, should_stop=None, max_steps=None):
    """Drive one SimplerEnv episode from an already-reset env. Returns (success, steps_taken)."""
    replan = int(getattr(policy, "replan_steps", 1))
    limit = int(max_steps) if max_steps else se.get_max_steps(env)
    policy.reset(instruction)
    ok, t = False, 0
    while t < limit:
        if should_stop and should_stop():
            break
        chunk = np.asarray(policy.predict_action_chunk(obs, instruction))
        term = trunc = False
        for i, a in enumerate(chunk[:replan]):
            obs, _r, term, trunc, info = env.step(np.asarray(a, dtype=np.float32))
            t += 1
            if se.is_success(info, term):
                ok = True
            if on_frame is not None:
                on_frame(se.get_image(env, obs), t, i == len(chunk[:replan]) - 1)
            if ok or term or trunc or (should_stop and should_stop()):
                break
        if ok or term or trunc:
            break
    return ok, t
