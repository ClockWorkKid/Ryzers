# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""RoboCasa scene wrapper for the interactive/sanity harness (model-agnostic).

Builds a single robosuite/RoboCasa env for a kitchen task and exposes the scene's native
instruction, the rendered 3-view observation + eef proprio, action stepping (zero-padding
the policy's 7-D delta into robosuite's full action_spec), and success checking. No policy
or model code here.
"""
import numpy as np

from sim_robocasa.robocasa_env import (
    ROBOCASA_ENV_RESOLUTION,
    TASK_MAX_STEPS,
    base2world,
    create_env,
    get_instruction,
    get_max_steps,
    render_obs,
)


class Scene:
    def __init__(self, task, seed=0, resolution=ROBOCASA_ENV_RESOLUTION):
        if task not in TASK_MAX_STEPS:
            raise ValueError(f"Unknown RoboCasa task: {task}")
        self.task = task
        self.seed = int(seed)
        self.resolution = int(resolution)
        self.max_steps = get_max_steps(task)

        self.env = create_env(env_name=task, seed=self.seed, render_onscreen=False,
                              camera_heights=self.resolution, camera_widths=self.resolution)
        self.env.reset()
        self._base2world = base2world(self.env)
        self.description = get_instruction(self.env)
        self.objects = self._list_objects()

    def _list_objects(self):
        """Best-effort object names for the viewport panel (cosmetic; any failure -> [])."""
        try:
            meta = self.env.get_ep_meta()
            objs = meta.get("object_cfgs", []) or []
            names = []
            for o in objs:
                n = (o.get("name") or o.get("obj_name") or "").replace("_", " ").strip()
                if n and n not in names:
                    names.append(n)
            return names[:12]
        except Exception:  # noqa: BLE001
            return []

    def reset(self):
        """Reset the env + re-anchor the base transform; returns the first obs dict."""
        self.env.reset()
        self._base2world = base2world(self.env)
        self.description = get_instruction(self.env)
        return self.observe()

    def observe(self):
        """Render the 3-view + proprio into the model-agnostic obs dict."""
        view, video, proprios = render_obs(self.env, self._base2world,
                                           camera_height=self.resolution, camera_width=self.resolution)
        return {"view": view, "video": video, "proprios": proprios,
                "instruction": self.description}

    def step(self, action):
        """Zero-pad the 7-D delta into robosuite's action_spec and step the env."""
        pad = np.zeros(self.env.action_spec[0].shape, dtype=np.float64)
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        pad[: min(7, a.shape[0])] = a[:7]
        self.env.step(pad)

    def check_success(self):
        try:
            return bool(self.env._check_success())
        except Exception:  # noqa: BLE001
            return False

    def view(self, obs):
        return obs["view"]

    def close(self):
        try:
            self.env.close()
        except Exception:  # noqa: BLE001
            pass


def build_scene(task, seed=0, resolution=ROBOCASA_ENV_RESOLUTION):
    return Scene(task, seed=seed, resolution=resolution)
