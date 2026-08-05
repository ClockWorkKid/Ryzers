#!/usr/bin/env python3
"""Apply gfx1151 inference speedups from the runtime optimization playbook."""

from __future__ import annotations

import os

import torch


def apply_gfx1151_speedups(*, disable_cudnn: bool | None = None, model=None) -> dict[str, bool]:
    """Return the active optimization flags after applying env-gated tweaks."""
    flags: dict[str, bool] = {}
    if disable_cudnn is None:
        disable_cudnn = os.environ.get("WAN22_DISABLE_CUDNN", "1") != "0"
    if disable_cudnn:
        torch.backends.cudnn.enabled = False
    flags["disable_cudnn"] = disable_cudnn

    from wan22_playbook_opts import apply_playbook_opts

    flags.update(apply_playbook_opts(model=model))
    return flags
