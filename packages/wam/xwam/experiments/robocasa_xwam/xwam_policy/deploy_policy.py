# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""X-WAM RoboCasa policy adapter (sim_robocasa.Policy seam, direct / in-process, no ZMQ).

Implements the shared `sim_robocasa.Policy` interface so X-WAM drives BOTH the RoboCasa
closed-loop benchmark and the interactive demos through one seam. Unlike RoboTwin (dual-arm
EE poses integrated + executed via mplib IK), RoboCasa is single-arm and its 7-D delta-EE
action is fed straight into robosuite's OSC_POSE composite controller by the harness -- so
this adapter is a thin wrapper: normalize the 3-view obs + proprio, run one in-process
`XWAMRunner.generate`, and return the denormalized `[Ta, 7]` delta chunk.

This mirrors upstream X-WAM's evaluation/robocasa_client.py request path (the frame
transforms it applies live in the observation build inside sim_robocasa.robocasa_env, kept
verbatim), with inference inlined via the shared DirectXWAM instead of the broker->server
ZMQ fabric. Selected at runtime with POLICY_FACTORY=deploy_policy:build_policy.

Directory note: this package lives under experiments/robocasa_xwam/ (NOT experiments/
robocasa/). The demo adds experiments/ to sys.path so the shared experiments/xwam_core.py
resolves; a directory literally named `robocasa` there becomes an implicit namespace package
that shadows the real pip `robocasa` (making `import robocasa` a no-op that registers zero
kitchen envs). The _xwam suffix avoids that collision.
"""
import os
import sys

import numpy as np

# xwam_core (shared in-process X-WAM inference) lives one level up under experiments/.
_EXP_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _EXP_DIR not in sys.path:
    sys.path.insert(0, _EXP_DIR)

from xwam_core import DirectXWAM, compute_seed  # noqa: E402

from sim_robocasa.policy import Policy  # noqa: E402


def _get(key, default=None):
    v = os.environ.get(key)
    if v is None or (isinstance(v, str) and v.strip().lower() in {"", "none", "null"}):
        return default
    return v


class XwamRoboCasaPolicy(Policy):
    """Wraps DirectXWAM as a sim_robocasa.Policy (single-arm 7-D delta-EE chunk)."""

    name = "xwam"

    def __init__(self, model: DirectXWAM, replan_steps: int, cfg: float):
        self.model = model
        self.replan_steps = int(replan_steps)
        self.num_steps_wait = 0
        self.cfg = float(cfg)
        self.episode_id = 0
        self.step_id = 0

    def reset(self, instruction):
        self.episode_id += 1
        self.step_id = 0

    def predict_action_chunk(self, obs, instruction):
        rgbs = np.asarray(obs["video"], dtype=np.float32)   # [V,H,W,3] in [-1,1]
        proprio = np.asarray(obs["proprios"], dtype=np.float64)  # [16]
        seed = compute_seed(0, self.episode_id, self.step_id)
        deltas = self.model.infer(rgbs, proprio, [instruction or ""], seed, cfg=self.cfg)
        self.step_id += 1
        return np.asarray(deltas, dtype=np.float32)  # [Ta, 7]


def build_policy():
    ckpt_root = str(_get("CKPT_ROOT", "/models/xwam/checkpoints"))
    exp = str(_get("EXP", "robocasa_sft"))
    exp_path = str(_get("EXP_PATH", os.path.join(ckpt_root, exp)))
    wan_ckpt = str(_get("WAN_CKPT_DIR", "/models/xwam/wan22_5b"))
    steps = str(_get("STEPS", "last"))
    denoise_steps = int(_get("DENOISE_STEPS", 50))
    action_denoise_steps = int(_get("ACTION_DENOISE_STEPS", 10))
    action_length = int(_get("ACTION_LENGTH", 32))
    replan_steps = int(_get("REPLAN_STEPS", action_length))
    cfg = float(_get("CFG", 0.0))

    model = DirectXWAM(
        exp_path=exp_path,
        wan_checkpoint_dir=wan_ckpt,
        steps=steps,
        denoise_steps=denoise_steps,
        action_denoise_steps=action_denoise_steps,
    )
    return XwamRoboCasaPolicy(model, replan_steps=replan_steps, cfg=cfg)
