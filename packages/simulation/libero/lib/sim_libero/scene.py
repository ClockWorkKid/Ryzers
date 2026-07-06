# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""LIBERO scene wrapper for the interactive/sanity harness (model-agnostic).

Builds a single OffScreenRenderEnv for a (suite, task_id) via the vendored glue and
exposes the scene's native instruction, initial state, and object list. No policy or
model code here.
"""
import numpy as np

from sim_libero.libero_env import (
    LIBERO_ENV_RESOLUTION,
    get_benchmark_dict,
    get_libero_env,
    get_libero_image,
)


class Scene:
    def __init__(self, suite, task_id, seed=1000, resolution=LIBERO_ENV_RESOLUTION):
        self.suite = suite
        self.task_id = int(task_id)
        self.seed = int(seed)
        self.resolution = int(resolution)

        benchmark_dict = get_benchmark_dict()
        task_suite = benchmark_dict[suite]()
        self.task = task_suite.get_task(self.task_id)
        self.init_states = task_suite.get_task_init_states(self.task_id)
        self.env, self.description = get_libero_env(self.task, self.resolution, self.seed)
        self.objects = self._list_objects()

    def _list_objects(self):
        try:
            names = list(getattr(self.task, "object_of_interest", []))
            if names:
                return [n.replace("_", " ") for n in names]
        except Exception:
            pass
        return []

    def reset(self):
        """Reset to the task's first initial state; returns the raw obs dict."""
        self.env.reset()
        idx = 0 if len(self.init_states) else None
        obs = self.env.set_init_state(self.init_states[idx]) if idx is not None else self.env.reset()
        return obs

    def view(self, obs):
        return get_libero_image(obs)

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass


def build_scene(suite, task_id, seed=1000, resolution=LIBERO_ENV_RESOLUTION):
    return Scene(suite, task_id, seed=seed, resolution=resolution)
