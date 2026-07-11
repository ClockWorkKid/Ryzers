# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Shared in-process X-WAM inference core for the Ryzer sim adapters.

Fuses the model-side pieces of upstream X-WAM (github.com/sharinka0715/X-WAM @ 72cfb86)
into one object so a Ryzer runs on a single Strix-Halo GPU without the upstream broker->
server->client ZMQ fabric:

  * inference        = `evaluation/policy_server.py` per-request block (resize+center-crop,
                       quantile proprio-norm, `XWAMRunner.generate(early_stop, run_depth=
                       False)`, quantile action denorm) -> `DirectXWAM`.
  * coordinate utils = `evaluation/X-WAM/deploy_policy.py` delta-EE integration + frame
                       transforms, kept verbatim (rule 2.1) for the RoboTwin EE path.

Consumers: the RoboTwin closed-loop `deploy_policy` (native seam), the RoboTwin interactive
`sim_robotwin.Policy` adapter, and the RoboCasa `sim_robocasa.Policy` adapter. RoboCasa is
single-arm and executes the denormalized 7-D delta directly (robosuite OSC_POSE), so it
uses only `DirectXWAM` + `compute_seed`; the transforms below are for RoboTwin's EE path.
"""
import hashlib
import os
import sys

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
