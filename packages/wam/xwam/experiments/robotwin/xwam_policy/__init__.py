# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""X-WAM RoboTwin policy plugin. RoboTwin's script/eval_policy.py does
`import xwam_policy` then reads get_model/eval/reset_model off the package."""
from .deploy_policy import get_model, eval, reset_model  # noqa: F401
