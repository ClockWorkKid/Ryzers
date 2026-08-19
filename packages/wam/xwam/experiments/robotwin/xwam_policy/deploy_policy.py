# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""X-WAM RoboTwin 2.0 closed-loop policy plugin (direct / in-process, no ZMQ).

This is the RoboTwin `deploy_policy` seam (`get_model` / `eval` / `reset_model`) that
`script/eval_policy.py` drives. It fuses two upstream X-WAM pieces
(github.com/sharinka0715/X-WAM @ 72cfb86) into a single in-process object so the Ryzer
runs on one Strix-Halo GPU without the upstream broker->server->client ZMQ fabric:

  * inference  = `evaluation/policy_server.py` per-request block (resize+center-crop,
                 quantile proprio-norm, `XWAMRunner.generate(early_stop, run_depth=False)`,
                 quantile action denorm) -> `DirectXWAM`.
  * EE control = `evaluation/X-WAM/deploy_policy.py` delta-EE integration + base/eef
                 coordinate transforms, kept verbatim, executed via
                 `TASK_ENV.take_action(..., action_type="ee")`.

The only project-specific changes are (a) inference is called in-process instead of over
ZMQ, and (b) a FastWAM-style action queue + `should_request_observation()` so the patched
RoboTwin harness can skip redundant `get_obs()` within a replan window. Closed-loop replan
cadence is `replan_steps` (re-anchors EE integration to the fresh end-effector pose each
replan; `replan_steps>=action chunk length` reproduces upstream execute-whole-chunk behaviour).
"""
import hashlib
import os
import sys
from collections import deque
from typing import Any, Dict, Optional

import numpy as np
import torch
from scipy.spatial.transform import Rotation

XWAM_REPO = os.environ.get("XWAM_REPO", "/repos/xwam")
if XWAM_REPO not in sys.path:
    sys.path.insert(0, XWAM_REPO)

# ---------------------------------------------------------------------------
# Coordinate transforms + delta-EE integration (verbatim from upstream
# evaluation/X-WAM/deploy_policy.py; rule 2.1 keep upstream logic unchanged).
# ---------------------------------------------------------------------------
# Base frame is redefined by rotating the frame +90 deg around its z axis.
BASE_COORD_XFORM = np.array(
    [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
)
# New eef axes: x' = z, z' = x, y' = -y (right-handed). Maps new eef frame -> old eef frame.
EEF_AXES_XFORM = np.array(
    [[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64
)


def compute_future_poses(initial_proprio: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Integrate [T,7] deltas (xyz + axisangle + gripper) onto an initial [8] pose
    (xyz + quat wxyz + gripper) -> absolute [T,8] poses in the global frame."""
    T = actions.shape[0]
    poses = np.zeros((T, 8), dtype=np.float64)
    pos = initial_proprio[:3].astype(np.float64).copy()
    quat_wxyz = initial_proprio[3:7].astype(np.float64)
    rot = Rotation.from_quat(quat_wxyz[[1, 2, 3, 0]])  # wxyz -> xyzw for scipy
    gripper = initial_proprio[7].astype(np.float64).copy()
    for i in range(T):
        d_pos = actions[i, :3]
        d_axisangle = actions[i, 3:6]
        pos = pos + d_pos
        dR = Rotation.from_rotvec(d_axisangle)
        rot = dR * rot
        poses[i, :3] = pos
        poses[i, 3:7] = rot.as_quat()[[3, 0, 1, 2]]
        gripper = gripper + actions[i, 6]
        poses[i, 7] = gripper
    return poses


def compute_seed(env_rank, rollout_id, step_id):
    """Deterministically derive a uint32 seed from (env_rank, rollout_id, step_id)."""
    key = f"{env_rank}_{rollout_id}_{step_id}"
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % (2**32)


# ---------------------------------------------------------------------------
# In-process inference (ported from evaluation/policy_server.py).
# ---------------------------------------------------------------------------
def _resize_and_center_crop_tensor(tensor, resized_shape, crop_ratio, depth=False):
    import torchvision.transforms.functional as TF

    B, V, _, _, _ = tensor.shape
    tensor = tensor.flatten(0, 1)
    tensor = TF.resize(tensor, size=resized_shape, interpolation=TF.InterpolationMode.BILINEAR, antialias=False)
    tensor = tensor.unflatten(0, (B, V))
    H, W = resized_shape
    crop_h = int(H * crop_ratio)
    crop_w = int(W * crop_ratio)
    top = (H - crop_h) // 2
    left = (W - crop_w) // 2
    out = []
    for b in range(B):
        out_b = []
        for v in range(V):
            img = TF.crop(tensor[b, v], top, left, crop_h, crop_w)
            interp = TF.InterpolationMode.NEAREST_EXACT if depth else TF.InterpolationMode.BILINEAR
            img = TF.resize(img, [H, W], interpolation=interp, antialias=False)
            out_b.append(img)
        out.append(torch.stack(out_b, dim=0))
    return torch.stack(out, dim=0)


def _build_statistics(config):
    """Quantile-normalization arrays from config.dataset.statistics (verbatim from policy_server)."""
    stats = config.dataset.statistics
    has_right_arm = "proprio_right_ee_xyz" in stats.q01
    state_q01 = list(stats.q01.proprio_left_ee_xyz) + [-1.0] * 4 + list(stats.q01.gripper_pos)
    state_q99 = list(stats.q99.proprio_left_ee_xyz) + [1.0] * 4 + list(stats.q99.gripper_pos)
    if has_right_arm:
        state_q01 += list(stats.q01.proprio_right_ee_xyz) + [-1.0] * 4 + list(stats.q01.gripper_pos)
        state_q99 += list(stats.q99.proprio_right_ee_xyz) + [1.0] * 4 + list(stats.q99.gripper_pos)
    else:
        state_q01 += [-1.0] * 8
        state_q99 += [1.0] * 8
    action_q01 = list(stats.q01.action_left_ee_xyz) + list(stats.q01.action_left_ee_axisangle) + list(stats.q01.gripper_action)
    action_q99 = list(stats.q99.action_left_ee_xyz) + list(stats.q99.action_left_ee_axisangle) + list(stats.q99.gripper_action)
    if has_right_arm:
        action_q01 += list(stats.q01.action_right_ee_xyz) + list(stats.q01.action_right_ee_axisangle) + list(stats.q01.gripper_action)
        action_q99 += list(stats.q99.action_right_ee_xyz) + list(stats.q99.action_right_ee_axisangle) + list(stats.q99.gripper_action)
    return np.array(state_q01), np.array(state_q99), np.array(action_q01), np.array(action_q99), has_right_arm


class DirectXWAM:
    """Loads XWAMRunner once and serves single-observation inference in-process."""

    def __init__(self, exp_path, wan_checkpoint_dir, steps="last", denoise_steps=50, action_denoise_steps=10):
        from omegaconf import OmegaConf
        import lightning as L
        from einops import rearrange
        from runners.xwam_runner import XWAMRunner

        self._rearrange = rearrange
        cfg_path = os.path.join(exp_path, "config.yaml")
        ckpt_path = os.path.join(exp_path, f"checkpoints/{steps}.ckpt/checkpoint/mp_rank_00_model_states.pt")
        for p in (cfg_path, ckpt_path):
            if not os.path.exists(p):
                raise FileNotFoundError(p)

        config = OmegaConf.load(cfg_path)
        config.sample_steps = denoise_steps
        config.use_decoupled_inference = action_denoise_steps > 0
        config.action_denoise_steps = action_denoise_steps
        config.action_num = config.dataset.frame_skip // config.dataset.action_skip
        if wan_checkpoint_dir is not None:
            config.wan_checkpoint_dir = wan_checkpoint_dir
        if config.get("wan_checkpoint_dir") is None:
            raise ValueError("Wan2.2-TI2V-5B checkpoint dir must be set (config or WAN_CKPT_DIR).")
        self.config = config
        self.video_size = list(config.dataset.video_size)

        (self.state_q01, self.state_q99, self.action_q01, self.action_q99,
         self.has_right_arm) = _build_statistics(config)
        self.action_dim = len(self.action_q01)

        L.seed_everything(int(config.seed), workers=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        model = XWAMRunner(config=config).cuda().bfloat16()
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["module"])
        model.eval()
        self.model = model

    @torch.inference_mode()
    def infer(self, rgbs_vhwc: np.ndarray, proprio_raw: np.ndarray, prompt, seed: int, cfg: float = 0.0):
        """rgbs_vhwc: [V,H,W,3] in [-1,1]; proprio_raw: [16] raw EE state.
        Returns denormalized delta-EE actions [Ta, action_dim]."""
        rgb = torch.from_numpy(rgbs_vhwc).bfloat16().unsqueeze(0).cuda()
        rgb = self._rearrange(rgb, "b v h w c -> b v c h w")
        rgb = _resize_and_center_crop_tensor(rgb, self.video_size, 0.95)

        proprio_norm = 2 * (proprio_raw - self.state_q01) / (self.state_q99 - self.state_q01) - 1
        if not self.has_right_arm:
            proprio_norm[8:] = 0.0
        proprio = torch.from_numpy(proprio_norm).bfloat16().unsqueeze(0).cuda()

        _, xt_actions, _, _ = self.model.generate(
            rgb, proprio, list(prompt), seeds=[seed], early_stop=True, cfg=cfg, run_depth=False
        )
        actions = xt_actions[0].float().cpu().numpy()
        actions = (actions[:, : self.action_dim] + 1) / 2 * (self.action_q99 - self.action_q01) + self.action_q01
        if not self.has_right_arm:
            actions[:, 6] *= -1.0  # invert gripper for single-arm (robocasa)
        return actions


# ---------------------------------------------------------------------------
# RoboTwin policy wrapper (queue + step; EE integration verbatim from upstream eval()).
# ---------------------------------------------------------------------------
CAMERA_IDS = ["head_camera", "left_camera", "right_camera"]


class XwamRoboTwinPolicy:
    def __init__(self, model: DirectXWAM, action_length: int, replan_steps: int, cfg: float):
        self.model = model
        self.action_length = int(action_length)
        self.replan_steps = int(max(1, min(replan_steps, action_length)))
        self.cfg = float(cfg)
        self.pending: deque[np.ndarray] = deque()
        self.episode_id = 0
        self.step_id = 0

    def should_request_observation(self) -> bool:
        return not self.pending

    def _obs_to_inputs(self, observation):
        rgbs = np.stack([observation["observation"][c]["rgb"] for c in CAMERA_IDS], axis=0)
        rgbs = rgbs.astype(np.float32) / 127.5 - 1.0  # [V,H,W,3]

        ep = observation["endpose"]
        left = self._arm_state(ep["left_endpose"], ep["left_gripper"])
        right = self._arm_state(ep["right_endpose"], ep["right_gripper"])
        robot_states = np.concatenate([left, right])  # [16], transformed frame
        return rgbs, robot_states, left, right

    @staticmethod
    def _arm_state(endpose, gripper):
        xyz = np.asarray(endpose[:3]) @ BASE_COORD_XFORM.T
        quat = np.array(endpose[3:])  # wxyz
        rot = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
        rot = BASE_COORD_XFORM @ rot @ EEF_AXES_XFORM
        quat = Rotation.from_matrix(rot).as_quat(canonical=True)[..., [3, 0, 1, 2]]
        return np.concatenate([xyz, quat, [gripper]])

    @staticmethod
    def _integrate_arm(arm_state, deltas):
        """Deltas [T,7] -> absolute EE poses [T,8] in sim frame (inverse of _arm_state xform)."""
        poses = compute_future_poses(arm_state, deltas)
        gripper = poses[:, 7:8]
        xyz = poses[:, :3] @ BASE_COORD_XFORM
        quat = poses[:, 3:7]
        mat = Rotation.from_quat(quat[..., [1, 2, 3, 0]]).as_matrix()
        mat = BASE_COORD_XFORM.T @ mat @ EEF_AXES_XFORM.T
        quat = Rotation.from_matrix(mat).as_quat(canonical=True)[..., [3, 0, 1, 2]]
        return xyz, quat, gripper

    def _plan(self, task_env, observation):
        rgbs, robot_states, left_arm, right_arm = self._obs_to_inputs(observation)
        instruction = task_env.get_instruction()
        seed = compute_seed(0, self.episode_id, self.step_id)
        deltas = self.model.infer(rgbs, robot_states, [instruction], seed, cfg=self.cfg)  # [Ta,14]

        left_xyz, left_quat, left_grip = self._integrate_arm(left_arm, deltas[:, 0:7])
        right_xyz, right_quat, right_grip = self._integrate_arm(right_arm, deltas[:, 7:14])
        actions = np.concatenate(
            [left_xyz, left_quat, left_grip, right_xyz, right_quat, right_grip], axis=1
        )  # [Ta,14] absolute EE
        n_exec = min(self.replan_steps, actions.shape[0])
        for i in range(n_exec):
            self.pending.append(actions[i])
        self.step_id += 1

    def step(self, task_env, observation):
        if not self.pending:
            if observation is None:
                raise ValueError("Observation required on replan (empty action queue).")
            self._plan(task_env, observation)
        if not self.pending:
            return
        action = self.pending.popleft()
        task_env.take_action(action, action_type="ee")

    def reset(self):
        self.pending.clear()
        self.episode_id += 1
        self.step_id = 0


# ---------------------------------------------------------------------------
# ROCm/mplib compat shim.
# ---------------------------------------------------------------------------
def _patch_mplib_planner():
    """Make RoboTwin's `MplibPlanner` robust + practical for X-WAM's EE/IK-planning path on ROCm
    (curobo unavailable). All fixes are contained to this process (shared simulation/robotwin base
    image is left untouched); FastWAM/AHA-WAM never hit this path (they drive `action_type="qpos"`):

    1. `constraint_pose` kwarg: RoboTwin's curobo->mplib replacement left `MplibPlanner.plan_*`
       without the `constraint_pose` kwarg that `CuroboPlanner` and `robot.py`'s EE path pass.
       In `take_action(action_type="ee")` it's always None (no constraint), so accept and ignore.
    2. exception safety: the underlying mplib planner *raises* (e.g. TOPP "Fail to parameterize
       path") on unreachable targets instead of returning a status; RoboTwin's `MplibPlanner`
       doesn't wrap it, so one bad target crashes the whole rollout. RoboTwin's EE branch already
       handles `{"status": "Fail"}` gracefully (holds the arm), so convert raises into that.
    3. planner mode: X-WAM emits dense, small per-step EE deltas. mplib RRT (`plan_pose`, up to
       10 retries x 5s) is impractically slow and flaky for this; `plan_screw` (interpolative, no
       RRT) is fast and well-suited to short pose increments. `XWAM_PLAN_MODE` (default `screw`)
       routes `plan_path` through `plan_screw`; set to `rrt` for the collision-aware path."""
    try:
        from envs.robot.planner import MplibPlanner
    except Exception:
        return
    plan_mode = os.environ.get("XWAM_PLAN_MODE", "screw").strip().lower()

    def _safe(orig):
        def wrapper(self, *args, constraint_pose=None, **kwargs):
            try:
                return orig(self, *args, **kwargs)
            except Exception:
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
                return self.plan_screw(now_qpos, target_pose, arms_tag=arms_tag, log=kwargs.get("log", False))
            return orig_plan_path(self, now_qpos, target_pose, *args, **kwargs)

        plan_path_wrapper._xwam_shim = True
        MplibPlanner.plan_path = plan_path_wrapper


# ---------------------------------------------------------------------------
# RoboTwin deploy_policy seam.
# ---------------------------------------------------------------------------
def _get(usr_args, key, default=None):
    v = usr_args.get(key, default)
    if isinstance(v, str) and v.strip().lower() in {"", "none", "null"}:
        return default
    return v


def get_model(usr_args: Dict[str, Any]):
    ckpt_root = str(_get(usr_args, "ckpt_root", os.environ.get("CKPT_ROOT", "/models/xwam/checkpoints")))
    exp = str(_get(usr_args, "exp", os.environ.get("EXP", "robotwin_sft")))
    exp_path = _get(usr_args, "exp_path", os.path.join(ckpt_root, exp))
    wan_ckpt = _get(usr_args, "wan_checkpoint_dir", os.environ.get("WAN_CKPT_DIR", "/models/xwam/wan22_5b"))
    steps = str(_get(usr_args, "steps", "last"))
    denoise_steps = int(_get(usr_args, "denoise_steps", 50))
    action_denoise_steps = int(_get(usr_args, "action_denoise_steps", 10))
    action_length = int(_get(usr_args, "action_length", 32))
    replan_steps = int(_get(usr_args, "replan_steps", action_length))
    cfg = float(_get(usr_args, "cfg", 0.0))

    _patch_mplib_planner()

    model = DirectXWAM(
        exp_path=str(exp_path),
        wan_checkpoint_dir=str(wan_ckpt),
        steps=steps,
        denoise_steps=denoise_steps,
        action_denoise_steps=action_denoise_steps,
    )
    return XwamRoboTwinPolicy(model, action_length=action_length, replan_steps=replan_steps, cfg=cfg)


def eval(TASK_ENV, model, observation: Optional[Dict[str, Any]]):
    model.step(TASK_ENV, observation)


def reset_model(model):
    model.reset()
