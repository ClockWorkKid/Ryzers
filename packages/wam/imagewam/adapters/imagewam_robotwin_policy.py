# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""ImageWAM RoboTwin policy adapter for the simulation/robotwin interactive harness.

Implements the model-agnostic `sim_robotwin.Policy` seam by wrapping the FLUX.2 ImageWAM
world-action model. Reuses the *validated* RoboTwin deploy policy from
experiments/robotwin/imagewam_policy/deploy_policy.py verbatim (config compose, model
instantiate + checkpoint load, processor/normalizer, 3-cam image/state builders and
`_infer_action_chunk`) so interactive rollouts match the closed-loop numbers.

Selected at runtime by the sim harness via
  POLICY_FACTORY=imagewam_robotwin_policy:build_policy

The parity-critical closed-loop path does NOT use this adapter: it runs RoboTwin's own
script/eval_policy.py against the same deploy_policy.py plugin (see demo_closedloop_robotwin.sh).
This wrapper only bridges that plugin's inference into the interactive Policy seam.

Env: CKPT, DATASET_STATS, MIXED_PRECISION (bf16), REPLAN_STEPS, NUM_INFERENCE_STEPS,
ACTION_HORIZON, FLUX2_VARIANT (4b), FLUX2_SRC, FLUX2_MODEL_PATH, FLUX2_AE_MODEL_PATH,
FLUX2_QWEN3_MODEL_SPEC, PROPRIO_DIM (14). Requires /repos/imagewam(+/src),
/repos/imagewam/experiments/robotwin, the flux2 src, /opt/sim and /opt/RoboTwin on PYTHONPATH.
"""
import os

import numpy as np

import experiments.robotwin.imagewam_policy.deploy_policy as D
from sim_robotwin.policy import Policy

VARIANT = os.environ.get("FLUX2_VARIANT", "4b")
DEFAULT_CKPT = f"/models/imagewam_release/robotwin/flux2_klein_{VARIANT}/model.pt"
DEFAULT_STATS = f"/models/imagewam_release/robotwin/flux2_klein_{VARIANT}/dataset_stats.json"


def _env(name, default=None):
    val = os.environ.get(name)
    return val if val else default


class ImageWAMRoboTwinPolicy(Policy):
    name = "imagewam"
    action_type = "qpos"

    def __init__(self, model):
        self.model = model  # deploy_policy.WorldActionRobotWinPolicy
        self.replan_steps = int(getattr(model, "replan_steps", 16))

    def reset(self, instruction):
        self.model.reset()

    def predict_action_chunk(self, obs, instruction):
        chunk = self.model._infer_action_chunk(observation=obs, instruction=instruction)
        return np.asarray(chunk, dtype=np.float32)


def build_policy():
    # ryzers passes optional knobs as empty strings; treat "" as unset.
    flux2_src = _env("FLUX2_SRC", "/repos/flux2")
    dit = _env("FLUX2_MODEL_PATH", "/models/flux2/FLUX.2-klein-base-4B/flux-2-klein-base-4b.safetensors")
    ae = _env("FLUX2_AE_MODEL_PATH", "/models/flux2/FLUX.2-klein-base-4B/ae.safetensors")
    qwen3 = _env("FLUX2_QWEN3_MODEL_SPEC", "Qwen/Qwen3-4B")

    usr_args = {
        "ckpt_setting": _env("CKPT", DEFAULT_CKPT),
        "dataset_stats_path": _env("DATASET_STATS", DEFAULT_STATS),
        "device": "cuda",  # ROCm torch reports as cuda; deploy_policy falls back to cpu if absent
        "mixed_precision": _env("MIXED_PRECISION", "bf16"),
        "sim_task": f"robotwin_flux2_klein_{VARIANT}_base_clean_imagewam",
        "robotwin_camera_layout": _env("ROBOTWIN_CAMERA_LAYOUT", "compact_288x256"),
        # Fill the flux2 model paths that the task config leaves as mandatory-missing (???),
        # mirroring how eval_robotwin_single.py forwards the resolved cfg.model.
        "model_overrides": {
            "flux2_src_path": flux2_src,
            "flux2_model_path": dit,
            "ae_model_path": ae,
            "variant": f"klein-base-{VARIANT}",
            "qwen3_model_spec": qwen3,
            "load_text_encoder": True,
            "pack_proprio_after_text": True,
            "proprio_dim": int(_env("PROPRIO_DIM", "14")),
        },
    }
    for key, env in (("replan_steps", "REPLAN_STEPS"),
                     ("num_inference_steps", "NUM_INFERENCE_STEPS"),
                     ("action_horizon", "ACTION_HORIZON")):
        val = os.environ.get(env)
        if val:
            usr_args[key] = val

    model = D.get_model(usr_args)
    print(f"[imagewam_robotwin_policy] model ready (ckpt={usr_args['ckpt_setting']}, "
          f"replan={getattr(model, 'replan_steps', '?')}, layout={usr_args['robotwin_camera_layout']})", flush=True)
    return ImageWAMRoboTwinPolicy(model)
