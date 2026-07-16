# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Model-agnostic RoboLab-AMD episode loop (chunk-replay).

Render the 3-view obs, predict a chunk, execute its first `replan_steps` rows (each applied
via the JOINT_POSITION composite controller + gripper), re-rendering each executed step
through an `on_frame` callback. Used by both the sanity runner and the interactive server.
Identical in shape to sim_robocasa.rollout; action semantics (joint space) live in scene.step.
"""


def run_episode(scene, policy, instruction, on_frame=None, should_stop=None, max_steps=None):
    """Run one episode; returns (success, num_model_steps).

    on_frame(view_rgb, step_idx, replanning) is called every executed step.
    should_stop() -> True aborts early (interactive Stop button).
    """
    replan_steps = int(getattr(policy, "replan_steps", 16))
    num_steps_wait = int(getattr(policy, "num_steps_wait", 0))
    if max_steps is None:
        max_steps = scene.max_steps

    obs = scene.reset()
    policy.reset(instruction)

    pending = []
    success = False
    model_steps = 0
    t = 0
    while t < max_steps + num_steps_wait:
        if should_stop is not None and should_stop():
            break

        if not pending:
            chunk = policy.predict_action_chunk(obs, instruction)
            pending = [row for row in chunk[:replan_steps]]
            model_steps += 1

        action = pending.pop(0)
        if on_frame is not None:
            on_frame(scene.view(obs), t, not pending)

        scene.step(action)
        t += 1
        if scene.check_success():
            success = True
            break
        obs = scene.observe()

    return success, model_steps
