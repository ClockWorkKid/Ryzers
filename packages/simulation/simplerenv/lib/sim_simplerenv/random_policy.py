# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Default no-model policy: small random deltas. Used when POLICY_FACTORY is unset."""
import numpy as np

from .policy import ACTION_DIM, Policy


class RandomPolicy(Policy):
    replan_steps = 5

    def __init__(self, scale=0.1, seed=0):
        self.scale = scale
        self.rng = np.random.default_rng(seed)

    def predict_action_chunk(self, obs, instruction):
        chunk = self.rng.uniform(-self.scale, self.scale, size=(self.replan_steps, ACTION_DIM))
        chunk[:, 6] = self.rng.choice([-1.0, 1.0], size=self.replan_steps)  # gripper open/close
        return chunk.astype(np.float32)


def build_policy():
    return RandomPolicy()
