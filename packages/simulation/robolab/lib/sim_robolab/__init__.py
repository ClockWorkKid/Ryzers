# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Model-agnostic RoboLab-AMD simulator harness.

AMD-native reimplementation of NVIDIA RoboLab-120 task semantics on the open robosuite/
MuJoCo backend (headless EGL, gfx1151). Ships the env glue (DROID Franka + Robotiq 2F-85
+ 3-camera rig + JOINT_POSITION control), a `Policy` seam, generic sanity/interactive
harnesses, and a built-in RandomPolicy. Any policy (Cosmos3-Nano-Policy, X-WAM, ...) drives
the sim by providing a `build_policy() -> Policy` factory selected via POLICY_FACTORY.

Does NOT reproduce upstream RoboLab's official numbers (different physics/renderer). See
docs/PILOT_PLAN.md.
"""
from sim_robolab.policy import Policy, load_policy

__all__ = ["Policy", "load_policy"]
