# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Model-agnostic RoboCasa simulator harness.

Ships the RoboCasa/robosuite env glue, a `Policy` seam, generic closed-loop/interactive
harnesses, and a built-in RandomPolicy. Any policy/model (X-WAM, ...) drives the sim by
providing a `build_policy() -> Policy` factory selected via POLICY_FACTORY.
"""
from sim_robocasa.policy import Policy, load_policy

__all__ = ["Policy", "load_policy"]
