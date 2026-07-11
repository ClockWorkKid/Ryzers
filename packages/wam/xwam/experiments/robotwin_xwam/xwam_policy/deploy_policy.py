# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""X-WAM RoboTwin interactive policy adapter (sim_robotwin.Policy seam, in-process).

This is the *interactive* sibling of the RoboTwin closed-loop seam
(experiments/robotwin/xwam_policy/deploy_policy.py). The closed-loop path plugs into
RoboTwin's own script/eval_policy.py (get_model/eval/reset_model) for parity; this one plugs
into the model-agnostic `sim_robotwin.Policy` interface so X-WAM drives the interactive HTTP
demos (sync + real-time) the exact same way the RoboCasa adapter drives its demos.

RoboTwin is dual-arm and EE-controlled: X-WAM emits [T,14] delta-EE (per arm: xyz + axisangle
+ gripper), which we integrate onto the current end-effector pose and hand to the harness as
absolute [T,16] EE poses (per arm: xyz + quat_wxyz + gripper). The harness executes each row
via TASK_ENV.take_action(row, action_type="ee"), where RoboTwin solves IK with its planner --
hence this adapter sets `action_type = "ee"`. The obs->inputs conversion, the +90deg base /
eef-axis frame transforms, and the delta integration are kept verbatim from upstream X-WAM
(github.com/sharinka0715/X-WAM @ 72cfb86, evaluation/X-WAM/deploy_policy.py) via the shared
xwam_core, matching the closed-loop seam exactly (rule 2.1: keep upstream logic unchanged).

Selected at runtime with POLICY_FACTORY=deploy_policy:build_policy.
"""
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

# xwam_core (shared in-process X-WAM inference + coordinate transforms) lives one level up
# under experiments/. Same resolution trick as the RoboCasa adapter.
_EXP_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _EXP_DIR not in sys.path:
    sys.path.insert(0, _EXP_DIR)

from xwam_core import (  # noqa: E402
    BASE_COORD_XFORM,
    EEF_AXES_XFORM,
    DirectXWAM,
    compute_future_poses,
    compute_seed,
)

from sim_robotwin.policy import Policy  # noqa: E402

# RoboTwin's three RGB views, in the order X-WAM's policy_server stacks them.
CAMERA_IDS = ["head_camera", "left_camera", "right_camera"]


def _get(key, default=None):
    v = os.environ.get(key)
    if v is None or (isinstance(v, str) and v.strip().lower() in {"", "none", "null"}):
        return default
    return v


def _arm_state(endpose, gripper):
    """Raw sim EE pose (xyz + quat_wxyz) + gripper -> X-WAM's transformed [8] arm state.

    Verbatim from upstream evaluation/X-WAM/deploy_policy.py (the +90deg base rotation and the
    x'=z / z'=x / y'=-y eef-axis remap). Mirrors the closed-loop seam's XwamRoboTwinPolicy.
    """
    xyz = np.asarray(endpose[:3]) @ BASE_COORD_XFORM.T
    quat = np.array(endpose[3:])  # wxyz
    rot = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
    rot = BASE_COORD_XFORM @ rot @ EEF_AXES_XFORM
    quat = Rotation.from_matrix(rot).as_quat(canonical=True)[..., [3, 0, 1, 2]]
    return np.concatenate([xyz, quat, [gripper]])


def _integrate_arm(arm_state, deltas):
    """Deltas [T,7] -> absolute EE poses [T,8] (xyz + quat_wxyz + gripper) in RoboTwin's sim
    frame (inverse of _arm_state's transform). Verbatim from the closed-loop seam."""
    poses = compute_future_poses(arm_state, deltas)
    gripper = poses[:, 7:8]
    xyz = poses[:, :3] @ BASE_COORD_XFORM
    quat = poses[:, 3:7]
    mat = Rotation.from_quat(quat[..., [1, 2, 3, 0]]).as_matrix()
    mat = BASE_COORD_XFORM.T @ mat @ EEF_AXES_XFORM.T
    quat = Rotation.from_matrix(mat).as_quat(canonical=True)[..., [3, 0, 1, 2]]
    return xyz, quat, gripper


def _obs_to_inputs(observation):
    """RoboTwin obs dict -> (rgbs [V,H,W,3] in [-1,1], robot_states [16], left [8], right [8])."""
    rgbs = np.stack([observation["observation"][c]["rgb"] for c in CAMERA_IDS], axis=0)
    rgbs = rgbs.astype(np.float32) / 127.5 - 1.0

    ep = observation["endpose"]
    left = _arm_state(ep["left_endpose"], ep["left_gripper"])
    right = _arm_state(ep["right_endpose"], ep["right_gripper"])
    robot_states = np.concatenate([left, right])  # [16], transformed frame
    return rgbs, robot_states, left, right


def _patch_mplib_planner():
    """Make RoboTwin's MplibPlanner robust + practical for X-WAM's EE/IK path on ROCm (curobo
    unavailable). Identical shim to the closed-loop seam; contained to this process, so the
    shared simulation/robotwin base image is left untouched:

    1. accept+ignore the `constraint_pose` kwarg RoboTwin's EE branch passes (None here);
    2. convert planner raises (e.g. TOPP "Fail to parameterize path" on unreachable targets)
       into RoboTwin's graceful {"status": "Fail"} (hold the arm) so one bad target doesn't
       crash the interactive rollout;
    3. route plan_path through the fast interpolative plan_screw (XWAM_PLAN_MODE=screw default),
       well-suited to X-WAM's dense small per-step EE deltas; set XWAM_PLAN_MODE=rrt for the
       collision-aware path.
    """
    try:
        from envs.robot.planner import MplibPlanner
    except Exception:  # noqa: BLE001
        return
    plan_mode = os.environ.get("XWAM_PLAN_MODE", "screw").strip().lower()

    def _safe(orig):
        def wrapper(self, *args, constraint_pose=None, **kwargs):
            try:
                return orig(self, *args, **kwargs)
            except Exception:  # noqa: BLE001
                return {"status": "Fail"}
        wrapper._xwam_shim = True
        return wrapper

    for name in ("plan_pose", "plan_screw"):
        fn = getattr(MplibPlanner, name, None)
        if fn is not None and not getattr(fn, "_xwam_shim", False):
            setattr(MplibPlanner, name, _safe(fn))

    plan_path = getattr(MplibPlanner, "plan_path", None)
    if plan_path is not None and not getattr(plan_path, "_xwam_shim", False):
        orig_plan_path = plan_path

        def plan_path_wrapper(self, now_qpos, target_pose, *args, constraint_pose=None, **kwargs):
            if plan_mode == "screw":
                arms_tag = kwargs.get("arms_tag")
                return self.plan_screw(now_qpos, target_pose, arms_tag=arms_tag,
                                       log=kwargs.get("log", False))
            return orig_plan_path(self, now_qpos, target_pose, *args, **kwargs)

        plan_path_wrapper._xwam_shim = True
        MplibPlanner.plan_path = plan_path_wrapper


class XwamRoboTwinInteractivePolicy(Policy):
    """Wraps DirectXWAM as a sim_robotwin.Policy that commands absolute EE poses."""

    name = "xwam"
    action_type = "ee"  # harness executes rows via take_action(..., action_type="ee")

    def __init__(self, model: DirectXWAM, replan_steps: int, cfg: float):
        self.model = model
        self.replan_steps = int(replan_steps)
        self.cfg = float(cfg)
        self.episode_id = 0
        self.step_id = 0
        self._patched = False

    def _ensure_mplib_patched(self):
        # Apply the mplib/IK shim lazily on first predict: the harness only chdirs into
        # ROBOTWIN_ROOT (making `envs.robot.planner` importable) once the first scene is built,
        # which happens AFTER build_policy(). Patching at build time would silently no-op.
        if not self._patched:
            _patch_mplib_planner()
            self._patched = True

    def reset(self, instruction):
        self.episode_id += 1
        self.step_id = 0

    def predict_action_chunk(self, obs, instruction):
        self._ensure_mplib_patched()
        rgbs, robot_states, left_arm, right_arm = _obs_to_inputs(obs)
        seed = compute_seed(0, self.episode_id, self.step_id)
        deltas = self.model.infer(rgbs, robot_states, [instruction or ""], seed, cfg=self.cfg)  # [Ta,14]

        left_xyz, left_quat, left_grip = _integrate_arm(left_arm, deltas[:, 0:7])
        right_xyz, right_quat, right_grip = _integrate_arm(right_arm, deltas[:, 7:14])
        actions = np.concatenate(
            [left_xyz, left_quat, left_grip, right_xyz, right_quat, right_grip], axis=1
        )  # [Ta,16] absolute EE poses in sim frame
        self.step_id += 1
        return np.asarray(actions, dtype=np.float64)


def build_policy():
    ckpt_root = str(_get("CKPT_ROOT", "/models/xwam/checkpoints"))
    exp = str(_get("EXP", "robotwin_sft"))
    exp_path = str(_get("EXP_PATH", os.path.join(ckpt_root, exp)))
    wan_ckpt = str(_get("WAN_CKPT_DIR", "/models/xwam/wan22_5b"))
    steps = str(_get("STEPS", "last"))
    denoise_steps = int(_get("DENOISE_STEPS", 50))
    action_denoise_steps = int(_get("ACTION_DENOISE_STEPS", 10))
    action_length = int(_get("ACTION_LENGTH", 32))
    replan_steps = int(_get("REPLAN_STEPS", 8))
    cfg = float(_get("CFG", 0.0))

    model = DirectXWAM(
        exp_path=exp_path,
        wan_checkpoint_dir=wan_ckpt,
        steps=steps,
        denoise_steps=denoise_steps,
        action_denoise_steps=action_denoise_steps,
    )
    return XwamRoboTwinInteractivePolicy(model, replan_steps=replan_steps, cfg=cfg)
