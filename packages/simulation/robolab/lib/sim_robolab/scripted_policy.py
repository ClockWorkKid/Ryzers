# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Deterministic scripted joint-space policy (no model) for controller validation.

Emits a known, open-loop absolute-joint trajectory anchored at the robot's reset configuration:
each arm joint follows a low-speed, phase-staggered sine, and the gripper follows a slow square
wave. Because it is emitted through the standard Policy seam (`predict_action_chunk`) and
executed via `scene.step` -> `abs_joints_to_action` -> the JOINT_POSITION composite controller,
it exercises the exact control path a real policy uses -- so measuring achieved-vs-commanded
joint angles validates the controller (gate G2) with no learned model.

Open-loop by design: the target is a function of the global step index only (anchored once at
the first observation), so achieved-vs-commanded error reflects controller tracking, not
feedback masking.
"""
import numpy as np

from sim_robolab.policy import Policy


class ScriptedJointPolicy(Policy):
    name = "scripted"
    replan_steps = 16

    def __init__(self, amplitude=0.2, period_steps=100, gripper_period_steps=100):
        self.amplitude = float(amplitude)      # rad, per-joint sine amplitude (free-space safe)
        self.period = int(period_steps)        # steps per sine period
        self.gperiod = int(gripper_period_steps)
        self._q0 = None
        self._t = 0

    def reset(self, instruction):
        self._q0 = None
        self._t = 0

    def target_at(self, t):
        """Absolute [7 joint targets + gripper] commanded at global step t (needs anchor set)."""
        phases = np.arange(7) * (2.0 * np.pi / 7.0)   # stagger joints for a legible overlay
        q = self._q0 + self.amplitude * np.sin(2.0 * np.pi * t / self.period + phases)
        gripper = 1.0 if np.sin(2.0 * np.pi * t / self.gperiod) >= 0.0 else -1.0
        return np.concatenate([q, [gripper]]).astype(np.float32)

    def predict_action_chunk(self, obs, instruction):
        if self._q0 is None:
            self._q0 = np.asarray(obs.get("qpos", np.zeros(7)),
                                  dtype=np.float64).reshape(-1)[:7].copy()
        rows = [self.target_at(self._t + i) for i in range(self.replan_steps)]
        self._t += self.replan_steps
        return np.stack(rows, axis=0)


def build_policy():
    return ScriptedJointPolicy()
