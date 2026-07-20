# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Headless iTHOR ObjectNav environment for AMD Strix Halo (gfx1151).

Adapted from AVDC_experiments experiment/benchmark_thor.py `ThorEnv` (rule 2.1: reuse upstream,
patch only what the platform needs). The single port change is making the ai2thor `Controller`
render headless via Vulkan (`platform=CloudRendering`, `gpu_device`) instead of the default path
that assumes an X server - everything else (ObjectNav reset-to-random-pose, discrete step set,
target-visible success, depth frame) is upstream behaviour.
"""
import os

import numpy as np


def _resolve_platform(name: str):
    """Map a THOR_PLATFORM string to an ai2thor platform class (None => ai2thor auto-detect)."""
    from ai2thor.platform import CloudRendering, Linux64  # noqa: F401

    table = {"CloudRendering": CloudRendering, "Linux64": Linux64, "": None, "auto": None}
    return table.get(name, CloudRendering)


class ThorEnv:
    """iTHOR ObjectNav env: navigate until `target` is visible. Depth comes from the sim."""

    def __init__(self, scene, target, seed=None, resolution=(64, 64), max_eplen=50,
                 rotate_step_degrees=45, visibility_distance=1.5,
                 platform=None, gpu_device=None, server_timeout=None):
        from ai2thor.controller import Controller

        if seed is None:
            seed = np.random.randint(int(1e6))
        if platform is None:
            platform = os.environ.get("THOR_PLATFORM", "CloudRendering")
        if gpu_device is None:
            gpu_device = int(os.environ.get("THOR_GPU_DEVICE", "0"))
        if server_timeout is None:
            server_timeout = float(os.environ.get("THOR_SERVER_TIMEOUT", "300"))

        kwargs = dict(
            scene=scene,
            rotateStepDegrees=rotate_step_degrees,
            visibilityDistance=visibility_distance,
            width=resolution[0],
            height=resolution[1],
            renderDepthImage=True,
            gpu_device=gpu_device,
            server_timeout=server_timeout,
        )
        plat = _resolve_platform(platform)
        if plat is not None:
            kwargs["platform"] = plat

        self.controller = Controller(**kwargs)
        self.max_eplen = max_eplen
        self.target = target
        self.objidx = None
        self.eplen = 0
        self.seed(seed)

    def seed(self, seed):
        self.rng = np.random.RandomState(seed)

    def reset(self):
        self.eplen = 0
        self._randomize_agent_pose()
        objidx = None
        for i, obj in enumerate(self.controller.last_event.metadata["objects"]):
            if self.target in obj["name"]:
                objidx = i
                break
        if objidx is None:
            raise ValueError(f"target {self.target!r} not found in scene objects")
        self.objidx = objidx
        return self._get_obs()

    def step(self, action):
        assert action in ("MoveAhead", "RotateRight", "RotateLeft", "Done"), action
        self.eplen += 1
        self.controller.step(action)
        done = self.eplen >= self.max_eplen or action == "Done"
        success = self._success()
        return self._get_obs(), success, done

    def close(self):
        try:
            self.controller.stop()
        except Exception:
            pass

    def _get_obs(self):
        frame = self.controller.last_event.frame            # (H, W, 3) uint8 RGB
        depth = self.controller.last_event.depth_frame       # (H, W) float32 metres
        return (frame, depth)

    def _randomize_agent_pose(self):
        self.controller.reset()
        positions = self.controller.step(action="GetReachablePositions").metadata["actionReturn"]
        position = self.rng.choice(positions)
        self.controller.step("Teleport", position=position)

    def _success(self):
        return bool(self.controller.last_event.metadata["objects"][self.objidx]["visible"])
