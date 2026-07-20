# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Built-in ScriptedPolicy: the default seam target, needs no model and no weights.

A trivial scan-and-advance ObjectNav heuristic (rotate to look around, occasionally step forward)
so `demo_sim_sanity.sh` exercises the full ai2thor CloudRendering + reset + step + render + video
path without any model. It is NOT a competent navigator - it just proves the simulator harness is
alive end to end on gfx1151.
"""
import os
from typing import List

import numpy as np

from sim_ithor.policy import Obs, Policy


class ScriptedPolicy(Policy):
    name = "scripted"

    def __init__(self, seed: int = 0, forward_every: int = 3, chunk: int = 4):
        self._rng = np.random.default_rng(int(seed))
        self._forward_every = int(forward_every)
        self._chunk = int(chunk)
        self._t = 0

    def reset(self, target: str) -> None:
        self._t = 0

    def plan(self, obs: Obs) -> List[str]:
        # Emit a short chunk: mostly rotate to scan the room, step forward periodically.
        actions: List[str] = []
        for _ in range(self._chunk):
            self._t += 1
            if self._t % self._forward_every == 0:
                actions.append("MoveAhead")
            else:
                actions.append("RotateRight" if self._rng.random() > 0.25 else "RotateLeft")
        return actions


def _env_int(name: str, default: int) -> int:
    # Treat unset OR empty ("") the same: config.yaml passes knobs as `-e SEED=${SEED:-}`, so an
    # unspecified var arrives as an empty string, which int("") would reject.
    val = os.environ.get(name, "")
    return int(val) if val not in ("", None) else int(default)


def build_policy() -> Policy:
    return ScriptedPolicy(
        seed=_env_int("SEED", 0),
        forward_every=_env_int("SCRIPTED_FORWARD_EVERY", 3),
        chunk=_env_int("SCRIPTED_CHUNK", 4),
    )
