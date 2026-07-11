# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Model-agnostic policy interface for the RoboCasa simulator harness.

A policy plugs into the closed-loop / interactive harness by implementing this ABC. The
harness owns the env, rendering, streaming and the episode loop; the policy only turns an
observation + instruction into an action chunk. Any model (X-WAM, ...) ships a factory
`build_policy() -> Policy` and is selected at runtime via the `POLICY_FACTORY=module:
function` env var (default: the built-in RandomPolicy).

RoboCasa control: robosuite's OSC_POSE composite controller consumes a 7-D per-step
end-effector delta [dx, dy, dz, drx, dry, drz, gripper]. The harness zero-pads each row
into the robot's full action_spec and calls `env.step`, so unlike RoboTwin there is no IK
/ motion-planning seam here -- the same Policy drives both closed-loop and interactive.
"""
import importlib
from abc import ABC, abstractmethod

from sim_robocasa.envutil import env_str


class Policy(ABC):
    """Turns (obs, instruction) into a [T, 7] delta-EE chunk. Harness executes it."""

    # RoboCasa control cadence knobs the harness reads (a model may override). RoboCasa's
    # upstream client executes a whole predicted chunk before re-observing, so the default
    # replan window equals the typical 32-step chunk.
    replan_steps = 32     # env steps executed per predicted chunk before replanning
    num_steps_wait = 0    # no-op settle steps at episode start (RoboCasa needs none)
    name = "policy"

    def reset(self, instruction):
        """Called once per episode before the first prediction (clear caches, etc.)."""

    @abstractmethod
    def predict_action_chunk(self, obs, instruction):
        """Return an ndarray of shape [T, 7] (dx,dy,dz, drx,dry,drz, gripper)."""

    def warmup(self, obs, instruction):
        """Optional one-time forward so the first real episode isn't stalled."""
        try:
            self.predict_action_chunk(obs, instruction)
        except Exception:  # noqa: BLE001 - warmup is best-effort
            pass


def load_policy():
    """Instantiate the policy named by POLICY_FACTORY=module:function (default RandomPolicy)."""
    spec = env_str("POLICY_FACTORY", "sim_robocasa.random_policy:build_policy")
    if ":" not in spec:
        raise ValueError(f"POLICY_FACTORY must be 'module:function', got {spec!r}")
    module_name, fn_name = spec.split(":", 1)
    factory = getattr(importlib.import_module(module_name), fn_name)
    policy = factory()
    if not isinstance(policy, Policy):
        raise TypeError(f"{spec} did not return a sim_robocasa.Policy (got {type(policy)})")
    return policy


__all__ = ["Policy", "load_policy"]
