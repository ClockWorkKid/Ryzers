# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""RoboLab-AMD scene wrapper for the interactive/sanity harness (model-agnostic).

Builds one robosuite env for a RoboLab task (DROID embodiment, JOINT_POSITION control),
exposes the instruction, the rendered 3-view observation + proprio, action stepping (maps
absolute joint targets -> the controller's normalised delta and steps), and success
(the MuJoCo-state object_in_container predicate). No policy/model code here.
"""
import numpy as np

from sim_robolab.robolab_env import (
    ENV_RESOLUTION,
    TASK_MAX_STEPS,
    abs_joints_to_action,
    arm_qpos,
    create_env,
    eef_pos,
    get_instruction,
    get_max_steps,
    gripper_openness,
    render_obs,
)


class Scene:
    def __init__(self, task, seed=0, resolution=ENV_RESOLUTION):
        if task not in TASK_MAX_STEPS:
            raise ValueError(f"Unknown RoboLab task: {task}")
        self.task = task
        self.seed = int(seed)
        self.resolution = int(resolution)
        self.max_steps = get_max_steps(task)

        self.env = create_env(env_name=task, seed=self.seed, render_onscreen=False,
                              camera_heights=self.resolution, camera_widths=self.resolution)
        self.env.reset()
        self.description = get_instruction(task)

    def reset(self):
        self.env.reset()
        return self.observe()

    def observe(self):
        view, video, proprios, qpos = render_obs(
            self.env, camera_height=self.resolution, camera_width=self.resolution
        )
        return {"view": view, "video": video, "proprios": proprios, "qpos": qpos,
                "instruction": self.description}

    def step(self, action):
        """Map [7 abs joint targets + gripper] -> robosuite action and step the env."""
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        target_qpos, gripper = a[:7], (a[7] if a.shape[0] > 7 else -1.0)
        self.env.step(abs_joints_to_action(self.env, target_qpos, gripper))

    def arm_qpos(self):
        """Live 7 arm joint angles (rad) -- for controller-tracking validation."""
        return arm_qpos(self.env)

    def gripper_openness(self):
        """Live gripper openness (~0 closed .. 1 open)."""
        return gripper_openness(self.env)

    def eef_pos(self):
        """Live end-effector world position [3]."""
        return eef_pos(self.env)

    def check_success(self):
        try:
            return bool(self.env._check_success())
        except Exception:  # noqa: BLE001
            return False

    def view(self, obs):
        return obs["view"]

    def close(self):
        try:
            self.env.close()
        except Exception:  # noqa: BLE001
            pass


def build_scene(task, seed=0, resolution=ENV_RESOLUTION):
    return Scene(task, seed=seed, resolution=resolution)
