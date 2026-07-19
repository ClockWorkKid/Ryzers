# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Model-agnostic MimicGen (robosuite/MuJoCo) simulator harness for AMD Strix Halo.

This package ships **only the simulator**: the robosuite/robomimic/NVlabs-mimicgen/MuJoCo
closed-loop stack (headless EGL/OSMesa) plus a small harness that exposes a model-agnostic
policy seam and generic closed-loop / sanity runners. It contains **no** policy or model
weights. Any VLA/WAM (VERA, ...) is layered on top as a separate ryzers image and selects a
policy at runtime through the seam (see README "Plug in your own policy").

The upstream reset-to-demo / success-tracking / video harness (`vera.env_runner.MimicgenRunner`)
is reused unchanged (rule 2.1); this harness only wraps it behind the `Policy` seam and the
`POLICY_FACTORY` selector, and provides the built-in `RandomPolicy` sanity path (no model).
"""
from sim_mimicgen.policy import Policy, load_policy

__all__ = ["Policy", "load_policy"]
