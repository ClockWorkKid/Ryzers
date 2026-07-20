# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""iTHOR ObjectNav task definitions (from AVDC_experiments experiment/benchmark_thor.py +
thor_exp/utils.py, reused unchanged per rule 2.1).

The AVDC iTHOR benchmark is object navigation across 4 scenes x 3 target objects each (12 tasks):
the agent starts at a random reachable pose and must navigate until the target object is visible.
"""
import numpy as np

# The 4 scenes x 3 targets used by upstream benchmark_thor.py (order preserved).
SCENE2TARGETS = {
    "FloorPlan1":   ["Toaster", "Spatula", "Bread"],        # kitchen
    "FloorPlan201": ["Painting", "Laptop", "Television"],   # living room
    "FloorPlan301": ["Blinds", "DeskLamp", "Pillow"],       # bedroom
    "FloorPlan401": ["Mirror", "ToiletPaper", "SoapBar"],   # bathroom
}

# Discrete ObjectNav action set (rotateStepDegrees=45 in the env). "Done" ends the episode.
ACTIONS = ["MoveAhead", "RotateRight", "RotateLeft", "Done"]


def all_tasks():
    """Flatten SCENE2TARGETS into an ordered list of (scene, target) pairs."""
    return [(scene, target) for scene, targets in SCENE2TARGETS.items() for target in targets]


def get_cmat(fov=90, resolution=(64, 64)):
    """Camera intrinsics for a square iTHOR frame (upstream thor_exp/utils.get_cmat)."""
    width, height = resolution
    assert width == height, "iTHOR uses square images"
    f = (1.0 / np.tan(np.deg2rad(fov) / 2)) * width / 2.0  # focal length
    cmat = np.eye(4)
    cmat[0, 0] = f
    cmat[1, 1] = f
    cmat[0, 2] = (width - 1) / 2.0
    cmat[1, 2] = (height - 1) / 2.0
    return cmat
