# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Model-agnostic RoboLab-AMD environment glue (robosuite/MuJoCo backend).

Builds a robosuite env for an AMD-native RoboLab task with the DROID-style embodiment
(Franka + Robotiq 2F-85) under a JOINT_POSITION composite controller (Cosmos3-Nano-Policy
emits joint targets), renders the 3-camera observation + proprio, and lists the shipped
tasks + per-task step limits. No policy/model code here.

Control contract (see sim_robolab.policy.Policy): a chunk row is
[q0..q6, gripper] = 7 absolute arm joint targets (rad) + gripper in [-1, 1]. `scene.step`
converts absolute targets -> the JointPositionController's normalised per-step delta.
"""
import copy
import os

import numpy as np

# 3-camera DROID-style rig. Pilot uses existing robosuite cameras as stand-ins for the
# over-shoulder externals; proper DROID over-shoulder poses are added before G4 (PILOT_PLAN).
CAMERA_NAMES = ["robot0_eye_in_hand", "agentview", "sideview"]
ENV_RESOLUTION = 256

TASK_MAX_STEPS = {
    "BananaInBowl": 500,   # upstream episode_length_s=50 @ ~10 policy Hz -> generous budget
}
TASKS = list(TASK_MAX_STEPS.keys())

# JointPositionController output scaling (rad per normalised unit); MUST match the value set
# in build_controller_config so scene.step's absolute->delta mapping is correct.
JOINT_OUTPUT_MAX = 0.2
# Joint PD stiffness. robosuite default (50) leaves ~0.045 rad tracking lag at 20 Hz; 200
# (crit. damped, within kp_limits<=300) converges within a control step so achieved==commanded.
JOINT_KP = 200.0

_TASK_CLASSES = None


def _task_classes():
    global _TASK_CLASSES
    if _TASK_CLASSES is None:
        from sim_robolab.tasks.banana_in_bowl import BananaInBowl, INSTRUCTION
        _TASK_CLASSES = {"BananaInBowl": (BananaInBowl, INSTRUCTION)}
    return _TASK_CLASSES


def get_max_steps(task_name):
    if task_name not in TASK_MAX_STEPS:
        raise ValueError(f"Unknown RoboLab task: {task_name}")
    return TASK_MAX_STEPS[task_name]


def list_envs():
    return [{"task": t, "max_steps": TASK_MAX_STEPS[t]} for t in TASKS]


def get_instruction(task_name):
    return _task_classes()[task_name][1]


def build_controller_config(robot="Panda", output_max=JOINT_OUTPUT_MAX, kp=JOINT_KP):
    """Default composite controller with the arm switched to JOINT_POSITION.

    Keeps the arm's nested `gripper` sub-config (robosuite asserts it exists), widens the joint
    delta range so absolute joint targets are reachable in a few steps, and stiffens the joint
    PD (kp) so achieved joints converge to commanded within one control step (gate G2).
    """
    from robosuite.controllers import (
        load_composite_controller_config,
        load_part_controller_config,
    )

    cfg = copy.deepcopy(load_composite_controller_config(controller=None, robot=robot))
    gripper_sub = cfg["body_parts"]["right"].get("gripper", {"type": "GRIP"})
    jp = load_part_controller_config(default_controller="JOINT_POSITION")
    jp["gripper"] = gripper_sub
    jp["output_max"] = output_max
    jp["output_min"] = -output_max
    jp["kp"] = float(kp)
    jp["damping_ratio"] = 1.0
    cfg["body_parts"]["right"] = jp
    return cfg


def create_env(
    env_name="BananaInBowl",
    robots="Panda",
    gripper_types="Robotiq85Gripper",
    camera_names=CAMERA_NAMES,
    camera_widths=ENV_RESOLUTION,
    camera_heights=ENV_RESOLUTION,
    seed=None,
    render_onscreen=False,
):
    """Build a robosuite env for `env_name` (DROID embodiment + JOINT_POSITION control)."""
    if env_name not in _task_classes():
        raise ValueError(f"Unknown RoboLab task: {env_name}")
    cls, _ = _task_classes()[env_name]
    controller_config = build_controller_config(
        robot=robots if isinstance(robots, str) else robots[0]
    )
    env = cls(
        robots=robots,
        gripper_types=gripper_types,
        controller_configs=controller_config,
        camera_names=camera_names,
        camera_widths=camera_widths,
        camera_heights=camera_heights,
        has_renderer=render_onscreen,
        has_offscreen_renderer=(not render_onscreen),
        use_camera_obs=False,
        use_object_obs=True,
        ignore_done=True,
        control_freq=20,
        seed=seed,
    )
    return env


def arm_qpos(env):
    """Current 7 arm joint angles (rad)."""
    return np.asarray(env.robots[0]._joint_positions, dtype=np.float64).reshape(-1)[:7]


def eef_pos(env):
    """Current end-effector (grasp site) world position [3]."""
    ctrl = env.robots[0].composite_controller
    arm = ctrl.arms[0]
    return np.array(ctrl.part_controllers[arm].ref_pos, dtype=np.float64)


def gripper_openness(env):
    """Current gripper openness in [0, 1] (0 = fully closed, 1 = fully open).

    The Robotiq85 driver joint (`gripper0_right_finger_joint`, range [0, 0.8]) reads ~0 when
    open and ~0.8 when closed, so raw `joint_pos/actuator_max` is a *closedness* fraction in
    [0, 1]. We map it to a clamped openness = 1 - closedness. (The earlier formula returned an
    unclamped, sign-inverted value -- e.g. >1 -- which fed the policy an out-of-distribution
    gripper proprio and stopped it from ever commanding a grasp.)
    """
    ctrl = env.robots[0].composite_controller
    gkey = list(ctrl.grippers.keys())[0] if getattr(ctrl, "grippers", None) else None
    if gkey is None:
        return 1.0
    gc = ctrl.part_controllers[gkey]
    amax = float(gc.actuator_max[0]) or 1.0
    closedness = float(np.array(gc.joint_pos[0:1])[0]) / amax
    return float(np.clip(1.0 - closedness, 0.0, 1.0))


def _euler_deg_to_wxyz(rx, ry, rz):
    from scipy.spatial.transform import Rotation as R
    q = R.from_euler("xyz", [rx, ry, rz], degrees=True).as_quat()  # xyzw
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def _env_floats(key, default=None):
    v = os.environ.get(key)
    if not v:
        return default
    return [float(x) for x in v.split(",")]


def apply_droid_cameras(env):
    """Nudge the 3 observation cameras toward the DROID rig distribution.

    Off by default; enabled with ROBOLAB_DROID_CAMS=1. Applied every render (cheap) so it
    survives hard resets. The wrist ZED-Mini stand-in (`robot0_eye_in_hand`) is the consistent,
    high-leverage view for grasping; the default robosuite mount frames the Robotiq fingers as a
    large near-field black mass. We pull the wrist cam back/up along the hand and pitch it so the
    fingers sit at the frame edge with the grasp region centred, and widen the exterior FoV from
    45 deg to ~DROID ZED-2 (~70 deg). All poses are overridable via env for cheap static tuning:
      WRIST_POS="x,y,z" (hand frame), WRIST_EULER="rx,ry,rz" deg (hand frame), WRIST_FOVY,
      EXT_FOVY, AGENT_POS/AGENT_EULER, SIDE_POS/SIDE_EULER.
    """
    if os.environ.get("ROBOLAB_DROID_CAMS", "0") not in ("1", "true", "True"):
        return
    from scipy.spatial.transform import Rotation as R
    m = env.sim.model
    data = env.sim.data
    wid = m.camera_name2id("robot0_eye_in_hand")

    # Wrist cam as a DROID-like over-the-gripper view: float it above-and-behind the grasp
    # site and look down at it. Computed in world frame (robust; hand-frame axes are hard to
    # reason about blind) then expressed in the parent hand body's frame. Tunables (metres):
    # WRIST_BACK (toward base, -x world), WRIST_UP (+z world), plus WRIST_FOVY.
    ctrl = env.robots[0].composite_controller
    arm = ctrl.arms[0]
    grasp = np.array(ctrl.part_controllers[arm].ref_pos, dtype=np.float64)
    back = float(os.environ.get("WRIST_BACK", "0.15"))
    up = float(os.environ.get("WRIST_UP", "0.22"))
    cam_world = grasp + np.array([-back, 0.0, up])
    look = grasp - cam_world
    look /= (np.linalg.norm(look) + 1e-9)
    right = np.cross(look, np.array([0.0, 0.0, 1.0]))
    right /= (np.linalg.norm(right) + 1e-9)
    cam_up = np.cross(right, look)
    R_world_cam = np.column_stack([right, cam_up, -look])  # cam looks along -z

    bid = int(m.cam_bodyid[wid])
    hand_pos = np.array(data.body_xpos[bid], dtype=np.float64)
    hand_mat = np.array(data.body_xmat[bid], dtype=np.float64).reshape(3, 3)
    pos_local = hand_mat.T @ (cam_world - hand_pos)
    R_local = hand_mat.T @ R_world_cam
    q = R.from_matrix(R_local).as_quat()  # xyzw
    m.cam_pos[wid] = pos_local
    m.cam_quat[wid] = np.array([q[3], q[0], q[1], q[2]])
    m.cam_fovy[wid] = float(os.environ.get("WRIST_FOVY", "70"))

    ext_fovy = float(os.environ.get("EXT_FOVY", "70"))

    def _set_ext(name, pos_key, eul_key):
        cid = m.camera_name2id(name)
        pos = _env_floats(pos_key)
        eul = _env_floats(eul_key)
        if pos is not None:
            m.cam_pos[cid] = np.asarray(pos, dtype=np.float64)
        if eul is not None:
            m.cam_quat[cid] = _euler_deg_to_wxyz(*eul)
        m.cam_fovy[cid] = ext_fovy

    _set_ext("agentview", "AGENT_POS", "AGENT_EULER")
    _set_ext("sideview", "SIDE_POS", "SIDE_EULER")


def abs_joints_to_action(env, target_qpos, gripper, output_max=JOINT_OUTPUT_MAX):
    """Map 7 absolute joint targets -> robosuite [7 norm delta + gripper] action row."""
    cur = arm_qpos(env)
    tgt = np.asarray(target_qpos, dtype=np.float64).reshape(-1)[:7]
    delta = np.clip((tgt - cur) / max(output_max, 1e-6), -1.0, 1.0)
    return np.concatenate([delta, [float(gripper)]]).astype(np.float64)


def render_obs(env, camera_names=CAMERA_NAMES, camera_height=ENV_RESOLUTION,
               camera_width=ENV_RESOLUTION):
    """Render the 3 cameras + proprio.

    Returns (rgbs_view, rgbs_norm, proprios, qpos):
      rgbs_view  : [H, V*W, 3] uint8 stitched viewport for display/video.
      rgbs_norm  : [V, H, W, 3] float32 in [-1, 1] model input.
      proprios   : [16] eef xyz(3) + quat wxyz(4) + gripper(1) + 8 zeros (DROID-style).
      qpos       : [7] arm joint angles (for joint-space policies).
    """
    from scipy.spatial.transform import Rotation as R

    apply_droid_cameras(env)
    rgbs = []
    for cam in camera_names:
        rgb = env.sim.render(height=camera_height, width=camera_width, camera_name=cam,
                             depth=False, segmentation=False)
        rgbs.append(rgb[::-1].copy())
    rgbs = np.stack(rgbs, axis=0)
    rgbs_view = rgbs.transpose(1, 0, 2, 3).reshape(camera_height, -1, 3)
    rgbs_norm = rgbs.astype(np.float32) / 127.5 - 1.0

    ctrl = env.robots[0].composite_controller
    arm = ctrl.arms[0]
    eef_pos = np.array(ctrl.part_controllers[arm].ref_pos)
    eef_mat = np.array(ctrl.part_controllers[arm].ref_ori_mat)
    quat = R.from_matrix(eef_mat).as_quat(canonical=True)[[3, 0, 1, 2]]  # xyzw -> wxyz

    proprios = np.concatenate([eef_pos, quat, [gripper_openness(env)], np.zeros(8)])
    return rgbs_view, rgbs_norm, proprios, arm_qpos(env)
