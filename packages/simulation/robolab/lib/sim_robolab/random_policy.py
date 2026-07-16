# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Built-in no-model policy for simulator sanity checks (joint space).

Emits a gentle random walk in Franka joint space anchored at the robot's current joint
positions (read from obs["qpos"]), so the arm visibly moves without leaving a sane
configuration. Proves the env renders and steps under JOINT_POSITION control with no
learned model. This is the default policy when POLICY_FACTORY is unset.
"""
import numpy as np

from sim_robolab.policy import Policy


class RandomPolicy(Policy):
    name = "random"
    replan_steps = 16

    def __init__(self, scale=0.05, seed=0):
        self.scale = scale          # rad per-step joint perturbation
        self.rng = np.random.default_rng(seed)
        self._target = None

    def reset(self, instruction):
        self._target = None

    def predict_action_chunk(self, obs, instruction):
        qpos = np.asarray(obs.get("qpos", np.zeros(7)), dtype=np.float32).reshape(-1)[:7]
        if self._target is None:
            self._target = qpos.copy()
        rows = []
        for _ in range(self.replan_steps):
            self._target = self._target + self.rng.uniform(-self.scale, self.scale, size=7).astype(np.float32)
            gripper = self.rng.choice([-1.0, 1.0])
            rows.append(np.concatenate([self._target, [gripper]]).astype(np.float32))
        return np.stack(rows, axis=0)


def build_policy():
    return RandomPolicy()
