# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Built-in no-model policy for simulator sanity checks.

Emits small random end-effector deltas so the arm visibly moves, proving the RoboCasa
env renders and steps without any learned model. This is the default policy when
POLICY_FACTORY is unset.
"""
import numpy as np

from sim_robocasa.policy import Policy


class RandomPolicy(Policy):
    name = "random"
    replan_steps = 16

    def __init__(self, scale=0.2, seed=0):
        self.scale = scale
        self.rng = np.random.default_rng(seed)

    def reset(self, instruction):
        pass

    def predict_action_chunk(self, obs, instruction):
        deltas = self.rng.uniform(-self.scale, self.scale, size=(self.replan_steps, 6))
        gripper = self.rng.choice([-1.0, 1.0], size=(self.replan_steps, 1))
        return np.concatenate([deltas, gripper], axis=1).astype(np.float32)


def build_policy():
    return RandomPolicy()
