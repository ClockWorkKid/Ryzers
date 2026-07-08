# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Model-agnostic SimplerEnv (real-to-sim) harness for the ryzers sim base.

SimplerEnv (github.com/simpler-env/SimplerEnv, CoRL 2024) evaluates real-world manipulation
policies in simulation on two embodiments -- Google Robot and WidowX+Bridge -- via a uniform
Gym API (``simpler_env.make(task)`` -> ``reset()`` / ``step(action)``). Actions are 7-D:
``[dx, dy, dz, drot_axangle(3), gripper]`` (the same shape as the LIBERO OSC-delta action),
so a VLA-JEPA adapter reuses the LIBERO preprocessing/action handling.

This package vendors the harness a policy needs: an env loader (build/reset/step/image/
instruction/horizon), a ``Policy`` seam (POLICY_FACTORY), a chunk-replay rollout loop, an
offscreen render/MP4 helper, a headless sanity runner, and command-driven + real-time
interactive servers. Model weights/code never live here -- a policy chains on top.
"""
from .policy import Policy, load_policy  # noqa: F401
