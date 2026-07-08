# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Env-var readers that treat an empty string as unset (ryzers passes ``VAR=${VAR:-}``)."""
import os


def env_str(name, default=None):
    v = os.environ.get(name)
    return v if v not in (None, "") else default


def env_int(name, default=None):
    v = env_str(name)
    return int(v) if v is not None else default


def env_float(name, default=None):
    v = env_str(name)
    return float(v) if v is not None else default
