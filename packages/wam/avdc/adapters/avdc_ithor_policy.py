# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""AVDC adapter for the model-agnostic iTHOR ObjectNav harness (sim_ithor).

Wires the AVDC video-diffusion policy behind the sim_ithor.Policy seam, reproducing the upstream
AVDC_experiments benchmark_thor.py plan-then-execute loop (rule 2.1):

    plan(frame, depth):
        vidplan = pred_video_thor(video_model, frame, target)      # imagine a short future clip
        _, _, color, flow, _ = pred_flow_frame(flow_model, vidplan)# dense optical flow between frames
        transforms = get_transforms_nav(depth, cmat, flow, ...)    # flow + depth -> rigid transforms
        actions = transforms2actions(transforms)                   # -> {MoveAhead,RotateL/R,Done}

Selected by the harness via POLICY_FACTORY=avdc_ithor_policy:build_policy. The env stepping, reset,
success tracking and video all live in the sim base; AVDC only turns observations into nav actions.
Same 3D-UNet model as the Meta-World AVDC image, so avdc_optim (fp16 + torch.compile) applies.
"""
import os
from typing import List

from sim_ithor.policy import Obs, Policy
from sim_ithor.tasks import get_cmat


def _get_video_model(ckpt_dir, milestone, sample_steps):
    from flowdiffusion.inference_utils import get_video_model_thor
    # Upstream uses get_video_model_thor(ckpts_dir, milestone); pass timestep when supported (DDIM).
    try:
        return get_video_model_thor(ckpts_dir=ckpt_dir, milestone=milestone, timestep=sample_steps)
    except TypeError:
        return get_video_model_thor(ckpts_dir=ckpt_dir, milestone=milestone)


class AVDCiThorPolicy(Policy):
    name = "avdc"

    def __init__(self, ckpt_dir=None, milestone=16, sample_steps=None, rgd_tfm_tries=8):
        import numpy as np  # noqa: F401

        # Upstream AVDC always runs from experiment/, and myutils resolves its assets CWD-relative
        # (pretrained/gmflow*.pth, `sys.path.append('core')`). Match that CWD so get_flow_model and
        # the rigid-transform helpers find their files (rule 2.1: adapt to upstream, don't patch it).
        exp_dir = os.environ.get("AVDC_EXP_DIR", "/repos/avdc/experiment")
        if os.path.isdir(exp_dir):
            os.chdir(exp_dir)

        from myutils import get_flow_model

        self.ckpt_dir = ckpt_dir or os.environ.get("CKPT_DIR", "/models/ithor")
        self.milestone = int(milestone)
        self.sample_steps = sample_steps
        self.rgd_tfm_tries = int(rgd_tfm_tries)
        self.target = None

        print(f"[avdc-ithor] video model: {self.ckpt_dir}/model-{self.milestone}.pt "
              f"(ddim steps={self.sample_steps})")
        self.video_model = _get_video_model(self.ckpt_dir, self.milestone, self.sample_steps)
        try:
            import avdc_optim
            avdc_optim.apply(self.video_model)   # env-gated fp16 / torch.compile (phase 6)
        except Exception as e:  # noqa: BLE001
            print(f"[avdc-ithor] avdc_optim skipped ({e})")
        print("[avdc-ithor] flow model: UniMatch GMFlow (pretrained)")
        self.flow_model = get_flow_model()
        self.cmat = get_cmat()[:3]

    def reset(self, target: str) -> None:
        self.target = target

    def plan(self, obs: Obs) -> List[str]:
        from myutils import pred_flow_frame, get_transforms_nav, transforms2actions
        from flowdiffusion.inference_utils import pred_video_thor

        frame, depth = obs
        vidplan = pred_video_thor(self.video_model, frame, self.target)
        _image1, _image2, _color, flow, _flow_b = pred_flow_frame(self.flow_model, vidplan)
        transforms = get_transforms_nav(depth, self.cmat, flow, rgd_tfm_tries=self.rgd_tfm_tries)
        actions = transforms2actions(transforms, verbose=False)
        return list(actions) if actions else ["Done"]


def _env_int(name: str, default: int) -> int:
    # config.yaml passes knobs as `-e VAR=${VAR:-}`, so an unspecified var arrives as "" (present but
    # empty), which int("") would reject; treat unset OR empty as the default.
    val = os.environ.get(name, "")
    return int(val) if val not in ("", None) else int(default)


def build_policy() -> Policy:
    ss = os.environ.get("SAMPLE_STEPS") or ""
    return AVDCiThorPolicy(
        ckpt_dir=os.environ.get("CKPT_DIR") or "/models/ithor",
        milestone=_env_int("MILESTONE", 16),
        sample_steps=(int(ss) if ss else None),
        rgd_tfm_tries=_env_int("RGD_TFM_TRIES", 8),
    )
