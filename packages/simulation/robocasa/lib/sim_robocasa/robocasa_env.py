# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Model-agnostic RoboCasa environment glue for the simulation/robocasa package.

Vendored from upstream X-WAM's evaluation/robocasa_client.py (github.com/sharinka0715/
X-WAM @ 72cfb86) so this simulator package has zero dependency on any policy/model repo.
Provides just the pieces a closed-loop or interactive harness needs: build a robosuite/
RoboCasa env for one of the 24 kitchen tasks, render the 3-view observation + the eef
proprio state, the per-task horizons, and the shipped task list.

The env build (`create_env`) and observation extraction (`render_obs`) are kept verbatim
from upstream (rule 2.1); only harness-facing helpers (`list_envs`, `get_max_steps`) are
added. RoboCasa executes a 7-D delta-EE action directly through robosuite's OSC_POSE
composite controller (no IK / motion planning), so the harness simply zero-pads the
policy's action into the full robosuite action_spec and steps the env.
"""
import contextlib
import logging
import os
import warnings

# Quiet the noisy third-party import chatter (not errors), matching sim_libero: robosuite
# logs a "no private macro file" WARNING; gym prints an "unmaintained / NumPy 2.0" notice
# straight to stderr at import time. Suppress robosuite by raising its logger to ERROR and
# gym by importing it once with stderr redirected. Real errors still propagate.
for _name in ("robosuite_logs", "robosuite"):
    logging.getLogger(_name).setLevel(logging.ERROR)
with contextlib.redirect_stderr(open(os.devnull, "w")):
    try:
        import gym  # noqa: F401
    except Exception:  # noqa: BLE001
        pass
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
from scipy.spatial.transform import Rotation as R

ROBOCASA_ENV_RESOLUTION = 256  # camera resolution used to render training data

CAMERA_NAMES = [
    "robot0_agentview_left",
    "robot0_agentview_right",
    "robot0_eye_in_hand",
]

# Robocasa -> pretrain: eef axes rotated +90 deg around z (verbatim from upstream client).
# R_pretrain = R_robocasa @ EEF_AXES_XFORM ; R_robocasa = R_pretrain @ EEF_AXES_XFORM.T
EEF_AXES_XFORM = np.array(
    [
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

# The 24 kitchen tasks + their per-task step limits (verbatim from robocasa_client.py).
TASK_MAX_STEPS = {
    # Pick and place tasks
    "PnPCounterToCab": 500,
    "PnPCabToCounter": 500,
    "PnPCounterToSink": 700,
    "PnPSinkToCounter": 500,
    "PnPCounterToMicrowave": 600,
    "PnPMicrowaveToCounter": 500,
    "PnPCounterToStove": 500,
    "PnPStoveToCounter": 500,
    # Door tasks
    "OpenSingleDoor": 500,
    "CloseSingleDoor": 500,
    "OpenDoubleDoor": 1000,
    "CloseDoubleDoor": 700,
    # Drawer tasks
    "OpenDrawer": 500,
    "CloseDrawer": 500,
    # Stove tasks
    "TurnOnStove": 500,
    "TurnOffStove": 500,
    # Sink tasks
    "TurnOnSinkFaucet": 500,
    "TurnOffSinkFaucet": 500,
    "TurnSinkSpout": 500,
    # Coffee tasks
    "CoffeeSetupMug": 600,
    "CoffeeServeMug": 600,
    "CoffeePressButton": 300,
    # Microwave tasks
    "TurnOnMicrowave": 500,
    "TurnOffMicrowave": 500,
}

TASKS = list(TASK_MAX_STEPS.keys())


def get_max_steps(task_name):
    if task_name not in TASK_MAX_STEPS:
        raise ValueError(f"Unknown RoboCasa task: {task_name}")
    return TASK_MAX_STEPS[task_name]


def list_envs():
    """Enumerate every shipped task for the env-picker dropdown (no env build)."""
    return [{"task": t, "max_steps": TASK_MAX_STEPS[t]} for t in TASKS]


def create_env(
    env_name,
    robots="PandaOmron",
    camera_names=CAMERA_NAMES,
    camera_widths=ROBOCASA_ENV_RESOLUTION,
    camera_heights=ROBOCASA_ENV_RESOLUTION,
    seed=None,
    render_onscreen=False,
    obj_instance_split="B",
    generative_textures=None,
    randomize_cameras=False,
    layout_and_style_ids=((1, 1), (2, 2), (4, 4), (6, 9), (7, 10)),
):
    """Build a robosuite/RoboCasa env for `env_name` (verbatim from upstream client)."""
    import robocasa  # noqa: F401  (registers the RoboCasa environments/robots)
    import robosuite
    from robosuite.controllers import load_composite_controller_config

    controller_config = load_composite_controller_config(
        controller=None,
        robot=robots if isinstance(robots, str) else robots[0],
    )

    env_kwargs = dict(
        env_name=env_name,
        robots=robots,
        controller_configs=controller_config,
        camera_names=camera_names,
        camera_widths=camera_widths,
        camera_heights=camera_heights,
        has_renderer=render_onscreen,
        has_offscreen_renderer=(not render_onscreen),
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=False,
        camera_depths=False,
        seed=seed,
        obj_instance_split=obj_instance_split,
        generative_textures=generative_textures,
        randomize_cameras=randomize_cameras,
        layout_and_style_ids=layout_and_style_ids,
        translucent_robot=False,
    )
    return robosuite.make(**env_kwargs)


def base2world(env):
    """4x4 base->world transform from the arm controller origin (call after reset)."""
    controller = env.robots[0].composite_controller
    base_pos, base_mat = (
        controller.part_controllers[controller.arms[0]].origin_pos,
        controller.part_controllers[controller.arms[0]].origin_ori,
    )
    m = np.eye(4)
    m[:3, :3] = base_mat
    m[:3, 3] = base_pos
    return m


def render_obs(env, base2world_mat, camera_names=CAMERA_NAMES,
               camera_height=ROBOCASA_ENV_RESOLUTION, camera_width=ROBOCASA_ENV_RESOLUTION):
    """Render the 3 cameras + eef proprio (verbatim from upstream client render_obs).

    Returns (rgbs_view, rgbs_norm, eef_states):
      rgbs_view  : [H, V*W, 3] uint8 stitched viewport for display/video.
      rgbs_norm  : [V, H, W, 3] float32 in [-1, 1] model input.
      eef_states : [16] = eef2base xyz(3) + quat wxyz(4) + gripper_openness(1) + 8 zeros.
    """
    rgbs = []
    for cam_name in camera_names:
        rgb = env.sim.render(
            height=camera_height, width=camera_width, camera_name=cam_name,
            depth=False, segmentation=False,
        )
        rgb = rgb[::-1].copy()
        rgbs.append(rgb)

    rgbs = np.stack(rgbs, axis=0)
    rgbs_view = rgbs.transpose(1, 0, 2, 3).reshape(camera_height, -1, 3)
    rgbs_norm = rgbs.astype(np.float32) / 127.5 - 1.0

    controller = env.robots[0].composite_controller
    eef_pos, eef_mat = (
        controller.part_controllers[controller.arms[0]].ref_pos,
        controller.part_controllers[controller.arms[0]].ref_ori_mat,
    )
    eef2world = np.eye(4)
    eef2world[:3, :3] = eef_mat
    eef2world[:3, 3] = eef_pos

    eef2base = np.linalg.inv(base2world_mat) @ eef2world
    eef2base_pos = eef2base[:3, 3]
    rot_mat = eef2base[:3, :3] @ EEF_AXES_XFORM  # robocasa -> pretrain
    rot_quat = R.from_matrix(rot_mat).as_quat(canonical=True)[..., [3, 0, 1, 2]]  # xyzw -> wxyz

    gripper_openness = (
        controller.part_controllers[list(controller.grippers.keys())[0]].joint_pos[0:1]
        / controller.part_controllers[list(controller.grippers.keys())[0]].actuator_max[0]
    )

    zero_padding = np.zeros(8)
    eef_states = np.concatenate([eef2base_pos, rot_quat, gripper_openness, zero_padding])
    return rgbs_view, rgbs_norm, eef_states


def get_instruction(env):
    """Natural-language task string for the current episode."""
    try:
        return env.get_ep_meta()["lang"]
    except Exception:  # noqa: BLE001
        return ""
