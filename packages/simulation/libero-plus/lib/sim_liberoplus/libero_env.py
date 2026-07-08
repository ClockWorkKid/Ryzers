# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Model-agnostic LIBERO-Plus environment glue for the simulation/libero-plus package.

LIBERO-Plus (github.com/sylvestf/LIBERO-plus) is a drop-in replacement for the `libero`
package: identical env/benchmark API, but each of the four suites is expanded into
thousands of *perturbation instances* (10,030 total) spanning 7 perturbation dimensions
and 5 difficulty levels. Evaluation is identical to LIBERO except that each task is run
with a single trial (its init state encodes one perturbation config).

This module vendors the same env helpers the harness needs (build env, pull the
agentview+wrist image, no-op action, per-suite horizons) and adds a loader for the
`task_classification.json` that maps every task to its perturbation category + difficulty,
so a closed-loop runner can slice the benchmark by dimension/difficulty and report
per-dimension robustness the way the LIBERO-Plus paper does.
"""
import contextlib
import json
import logging
import os
import pathlib
import warnings

# Quiet the noisy third-party import chatter (not errors). robosuite logs a "no private
# macro file" WARNING via its logger; gym prints its "unmaintained / NumPy 2.0" notice
# straight to stderr at import time (NOT through warnings, so a filter can't catch it).
# Suppress robosuite by raising its logger to ERROR, and gym by eagerly importing it once
# with stderr redirected -- later imports (by libero/robosuite) hit the module cache and
# stay quiet. Real errors still propagate (logger level is ERROR, not CRITICAL).
for _name in ("robosuite_logs", "robosuite"):
    logging.getLogger(_name).setLevel(logging.ERROR)
with contextlib.redirect_stderr(open(os.devnull, "w")):
    try:
        import gym  # noqa: F401
    except Exception:  # noqa: BLE001
        pass
# Set the warnings filter AFTER importing gym: gym resets the warnings registry on import,
# which would otherwise wipe this filter and let robosuite's deprecated-.warn() notice leak.
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
from libero.libero import benchmark as _benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data

# LIBERO-Plus expands the four standard suites (libero_90 is not perturbed).
SUITES = ["libero_object", "libero_goal", "libero_spatial", "libero_10"]

# The 7 perturbation dimensions, as named in task_classification.json.
PERTURBATION_CATEGORIES = [
    "Camera Viewpoints",
    "Robot Initial States",
    "Language Instructions",
    "Light Conditions",
    "Background Textures",
    "Sensor Noise",
    "Objects Layout",
]

_SUITE_MAX_STEPS = {
    "libero_spatial": 400,
    "libero_object": 400,
    "libero_goal": 400,
    "libero_10": 700,
    "libero_90": 700,
}


def get_max_steps(task_suite_name):
    if task_suite_name not in _SUITE_MAX_STEPS:
        raise ValueError(f"Unknown task suite: {task_suite_name}")
    return _SUITE_MAX_STEPS[task_suite_name]


def get_benchmark_dict():
    return _benchmark.get_benchmark_dict()


def list_envs():
    """Enumerate every shipped (suite, task_id, description) for the env-picker dropdown.

    Cheap: reads task metadata from the benchmark registry without building any sim env.
    Note: LIBERO-Plus expands each suite into thousands of perturbation instances, so this
    can return a large list (up to ~10,030) -- the picker is intended for manual browsing.
    """
    out = []
    bench = get_benchmark_dict()
    for suite in SUITES:
        try:
            task_suite = bench[suite]()
            n_tasks = int(getattr(task_suite, "n_tasks", 0))
            for task_id in range(n_tasks):
                out.append({
                    "suite": suite,
                    "task_id": task_id,
                    "description": task_suite.get_task(task_id).language,
                })
        except Exception:  # noqa: BLE001 - skip a suite that fails to enumerate
            continue
    return out


def get_libero_env(task, resolution, seed):
    """Initialize a single OffScreenRenderEnv for a task; returns (env, description)."""
    task_description = task.language
    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    )
    # LIBERO-Plus's env_wrapper does substring checks on bddl_file_name (e.g. "_view_" in
    # name) to detect perturbation markers, so it must be a str, not a PosixPath.
    env = OffScreenRenderEnv(
        bddl_file_name=str(task_bddl_file),
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)  # seed affects object positions even with a fixed initial state
    return env, task_description


def get_libero_dummy_action():
    """No-op action (open gripper) used to settle the sim while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]


def get_libero_image(obs):
    """Extract + preprocess the agentview and wrist images (rotate 180 to match training)."""
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    return {"image": img, "wrist_image": wrist_img}


# --------------------------------------------------------------------------------------
# LIBERO-Plus perturbation classification (dimension + difficulty per task instance).
# --------------------------------------------------------------------------------------
def get_classification_path():
    """Locate task_classification.json shipped by the installed LIBERO-Plus package.

    Honors LIBEROPLUS_CLASSIFICATION as an explicit override; otherwise resolves it next
    to the installed `libero.libero` package (…/libero/libero/benchmark/…).
    """
    override = os.environ.get("LIBEROPLUS_CLASSIFICATION")
    if override:
        return pathlib.Path(override)
    import libero.libero as _ll

    return pathlib.Path(_ll.__file__).parent / "benchmark" / "task_classification.json"


def load_task_classification(path=None):
    """Return {suite: [{'id','name','category','difficulty_level'}, ...]} for all suites."""
    path = pathlib.Path(path) if path is not None else get_classification_path()
    if not path.exists():
        raise FileNotFoundError(
            f"task_classification.json not found at {path}. Is LIBERO-Plus installed with "
            f"its assets? Set LIBEROPLUS_CLASSIFICATION to point at it."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_task_ids(task_suite, entries):
    """Map classification entries to 0-based task_ids for a built benchmark suite.

    Prefers matching by task `name` (robust to any id indexing convention); falls back to
    `id - 1` (the json ids are 1-based) when a name is not present in the suite. Returns a
    list of (task_id, entry) pairs in the entries' order.
    """
    names = task_suite.get_task_names()
    name_to_idx = {n: i for i, n in enumerate(names)}
    out = []
    for e in entries:
        tid = name_to_idx.get(e.get("name"))
        if tid is None:
            cand = int(e.get("id", 0)) - 1
            if 0 <= cand < len(names):
                tid = cand
        if tid is not None:
            out.append((tid, e))
    return out
