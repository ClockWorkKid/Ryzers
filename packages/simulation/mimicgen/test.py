# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""MimicGen simulator base import sign-of-life (no env build, no render, no model).

Verifies ROCm torch is intact and the simulator/harness stack imports: robosuite/robomimic/
NVlabs-mimicgen/MuJoCo, the upstream MimicgenRunner (reset-to-demo eval harness), the websocket
client seam, and the model-agnostic sim_mimicgen harness (Policy seam + RandomPolicy)."""
import torch

assert torch.version.hip, "torch is not a ROCm build: " + torch.__version__

import robosuite  # noqa: E402
import robomimic  # noqa: E402
import mimicgen  # noqa: E402
import mujoco  # noqa: E402
from mimicgen.utils.robomimic_utils import create_env  # noqa: E402,F401

# Upstream eval harness (reused, rule 2.1) + the wire-protocol seam.
from vera.env_runner.mimicgen_runner import MimicgenRunner, MimicgenRunnerCfg  # noqa: E402,F401
from vera.server.protocol.websocket_policy_client import WebsocketClientPolicy  # noqa: E402,F401

# Model-agnostic harness.
from sim_mimicgen import Policy, load_policy  # noqa: E402
from sim_mimicgen.random_policy import RandomPolicy, build_policy  # noqa: E402

p = build_policy()
assert isinstance(p, Policy)

print("OK sim-mimicgen | torch", torch.__version__, "hip", torch.version.hip,
      "| robosuite", robosuite.__version__, "| mujoco", mujoco.__version__,
      "| default policy:", p.name)
