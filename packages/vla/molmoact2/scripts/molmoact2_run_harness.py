# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Apply the optional MolmoAct2 cross-embodiment swap, then hand off to the shared
simulation/libero harness entrypoint named by SIM_HARNESS_MODULE.

The shared harness is model-agnostic and must not be edited, so the arm swap
(EMBODIMENT=ur5e/xarm6) is installed here -- before the harness imports/builds any
scene -- by patching sim_libero.libero_env.OffScreenRenderEnv (see embodiment.py).
No-op for the default Panda. embodiment.py is resolved from this script's dir
(/ryzers), which also holds assets/xarm6.
"""
import os
import runpy

import embodiment

robot = embodiment.apply_from_env()
if robot:
    print(f"[embodiment] LIBERO arm swapped -> {robot}", flush=True)

runpy.run_module(os.environ.get("SIM_HARNESS_MODULE", "sim_libero.sanity"), run_name="__main__")
