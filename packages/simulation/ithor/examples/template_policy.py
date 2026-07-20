# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""COPY-ME: annotated skeleton for plugging a model into the iTHOR ObjectNav harness.

Selected at runtime via POLICY_FACTORY=module:function. iTHOR ObjectNav is plan-then-execute, so
implement `plan(obs) -> list[str]`: given the current (frame, depth), return a segment of discrete
actions from {MoveAhead, RotateLeft, RotateRight, Done}. Return ["Done"] (or []) to end. This is
exactly how AVDC drives it (imagine a future video, convert to moves, execute, re-plan) - see the
AVDC consumer's adapter (packages/wam/avdc-ithor/adapters/avdc_ithor_policy.py).

    POLICY_FACTORY=template_policy:build_policy ryzers run --name <yourmodel>-ithor \\
      SCENE=FloorPlan1 TARGET=Toaster /ryzers/demos/demo_sim_sanity.sh
"""
from typing import List

from sim_ithor.policy import Obs, Policy


class TemplatePolicy(Policy):
    name = "template"

    def __init__(self):
        # TODO: load your model / weights here.
        self._target = None

    def reset(self, target: str) -> None:
        """Called once per episode with the target object name (e.g. 'Toaster')."""
        self._target = target

    def plan(self, obs: Obs) -> List[str]:
        # obs = (frame, depth): frame (H,W,3 uint8 RGB), depth (H,W float32 metres).
        # TODO: run your model and return the next segment of discrete nav actions.
        frame, depth = obs  # noqa: F841
        return ["RotateRight"]


def build_policy() -> Policy:
    return TemplatePolicy()
