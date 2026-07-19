# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Built-in RandomPolicy: the default seam target, needs no model and no policy server.

Emits small random OSC_POSE deltas so a `demo_sim_sanity.sh` rollout exercises the full
robosuite/MuJoCo + reset-to-demo + render + video path without any model. Not a controller:
it will not solve tasks, it just proves the simulator harness is alive end to end.
"""
import numpy as np

from sim_mimicgen.policy import Policy, PolicyObservation, PolicyOutput

# MimicGen tasks use robosuite's OSC_POSE composite controller: a 7-D per-step action
# [dx, dy, dz, drx, dry, drz, gripper].
_ACTION_DIM = 7


class RandomPolicy(Policy):
    name = "random"
    cfg = None
    device = None

    def __init__(self, action_dim: int = _ACTION_DIM, scale: float = 0.1, seed: int = 0):
        self._dim = int(action_dim)
        self._scale = float(scale)
        self._rng = np.random.default_rng(int(seed))

    def reset(self) -> None:
        pass

    def predict_action(self, obs: PolicyObservation) -> PolicyOutput:
        del obs
        a = self._rng.uniform(-self._scale, self._scale, size=self._dim).astype(np.float32)
        a[-1] = 1.0 if self._rng.random() > 0.5 else -1.0  # open/close gripper
        return PolicyOutput(action=a[None, :], info=None)  # (1, D) for the env step


def build_policy() -> Policy:
    import os

    return RandomPolicy(
        action_dim=int(os.environ.get("RANDOM_ACTION_DIM", _ACTION_DIM)),
        scale=float(os.environ.get("RANDOM_ACTION_SCALE", 0.1)),
        seed=int(os.environ.get("SEED", 0)),
    )
