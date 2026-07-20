# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Model-agnostic policy seam for the iTHOR ObjectNav harness.

The harness owns the env, headless rendering, reset-to-random-pose, success tracking and the
episode loop (`sim_ithor.rollout.run_rollout`). A policy only turns observations into navigation
actions. Any model (AVDC, ...) ships a factory `build_policy() -> Policy` and is selected at
runtime with `POLICY_FACTORY=module:function` (default: the built-in `ScriptedPolicy`, which needs
no model / no weights).

Seam contract. iTHOR ObjectNav is plan-then-execute (this is how AVDC drives it: imagine a short
future video from the current frame + target, convert it to a sequence of discrete moves, execute
them, then re-plan). So the seam is chunked, not per-step:

    reset(target: str) -> None                          # once per episode
    plan(obs: (frame, depth)) -> list[str]              # a segment of {MoveAhead,RotateLeft,
                                                         #  RotateRight,Done}; [] or ["Done"] ends

A per-step policy simply returns a single-action list. `obs` is the tuple returned by ThorEnv:
`frame` (H,W,3 uint8 RGB), `depth` (H,W float32 metres).
"""
import importlib
import os
from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np

Obs = Tuple[np.ndarray, np.ndarray]


class Policy(ABC):
    """Model-agnostic ObjectNav seam (plan-then-execute)."""

    name = "policy"

    def reset(self, target: str) -> None:
        """Called once per episode with the target object name."""

    @abstractmethod
    def plan(self, obs: Obs) -> List[str]:
        """Return the next segment of discrete actions to execute (may be a single action)."""
        raise NotImplementedError


def load_policy() -> Policy:
    """Instantiate the policy named by POLICY_FACTORY=module:function (default ScriptedPolicy)."""
    spec = os.environ.get("POLICY_FACTORY") or "sim_ithor.scripted_policy:build_policy"
    if ":" not in spec:
        raise ValueError(f"POLICY_FACTORY must be 'module:function', got {spec!r}")
    module_name, fn_name = spec.split(":", 1)
    factory = getattr(importlib.import_module(module_name), fn_name)
    policy = factory()
    if not isinstance(policy, Policy):
        raise TypeError(f"{spec} did not return a sim_ithor.Policy (got {type(policy)})")
    return policy


__all__ = ["Policy", "load_policy", "Obs"]
