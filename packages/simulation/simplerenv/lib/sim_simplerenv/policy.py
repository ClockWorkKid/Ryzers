# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Policy seam: (obs, instruction) -> [T, 7] action chunk, selected by POLICY_FACTORY.

SimplerEnv actions are 7-D ``[dx, dy, dz, drot_axangle(3), gripper]`` in the maniskill2
world frame -- identical in shape to the LIBERO OSC-delta action, so a VLA-JEPA adapter can
reuse its LIBERO action handling. A policy returns a chunk of such rows; the rollout loop
(and interactive servers) replay ``replan_steps`` rows before requesting the next chunk.
"""
import importlib
from abc import ABC, abstractmethod

from .envutil import env_str

ACTION_DIM = 7


class Policy(ABC):
    """Base class every policy implements. Defaults match the LIBERO adapter cadence."""

    replan_steps = 5      # env steps executed per predicted chunk before replanning
    num_steps_wait = 0    # settle steps before acting (SimplerEnv resets to a stable pose)

    def reset(self, instruction):
        """Optional per-episode reset hook (e.g. clear the model's obs/action history)."""

    @abstractmethod
    def predict_action_chunk(self, obs, instruction):
        """Return an ndarray of shape [T, 7] (delta xyz + delta rot axangle + gripper)."""

    def warmup(self, obs, instruction):
        """Best-effort one-shot inference so the first real episode isn't stalled by JIT."""
        try:
            self.reset(instruction)
            self.predict_action_chunk(obs, instruction or "pick up the object")
        except Exception:  # noqa: BLE001 - warmup is best-effort
            pass


def load_policy():
    """Instantiate the policy named by POLICY_FACTORY=module:function (default RandomPolicy)."""
    spec = env_str("POLICY_FACTORY", "sim_simplerenv.random_policy:build_policy")
    module_name, fn_name = spec.split(":", 1)
    factory = getattr(importlib.import_module(module_name), fn_name)
    policy = factory()
    if not isinstance(policy, Policy):
        raise TypeError(f"{spec} returned {type(policy)!r}, not a sim_simplerenv.Policy")
    return policy
