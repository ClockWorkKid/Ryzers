# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Model-agnostic AI2-THOR (iTHOR) ObjectNav simulator harness for AMD Strix Halo (gfx1151).

Ships **only the simulator**: ai2thor + its Unity CloudRendering build (headless Vulkan) plus a
small harness that exposes a model-agnostic policy seam (`Policy` + `POLICY_FACTORY`), a `ThorEnv`
ObjectNav wrapper, and a generic rollout / sanity runner with a built-in `ScriptedPolicy`. It
contains **no** policy or model weights. A video/VLA model (AVDC, ...) is layered on top as a
separate ryzers image and selects a policy at runtime through the seam (see README).

The upstream ObjectNav task (env + scene2targets + camera intrinsics) is reused from
AVDC_experiments benchmark_thor.py / thor_exp (rule 2.1); the only port change is headless Vulkan
rendering (`platform=CloudRendering`).
"""
from sim_ithor.policy import Policy, load_policy

__all__ = ["Policy", "load_policy"]
