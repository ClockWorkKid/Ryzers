# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""VLA-JEPA LIBERO policy adapter for the simulation/libero harness.

Implements the model-agnostic ``sim_libero.Policy`` seam by wrapping VLA-JEPA (the starVLA
``baseframework``). Reuses this package's *validated* open-loop inference path verbatim
(``scripts/model_smoke.py`` checkpoint resolve + config repoint, ``scripts/openloop_replay.py``
un-normalization and gripper remap) so interactive / closed-loop rollouts match the shipped
open-loop numbers. Preprocessing follows upstream ``examples/LIBERO/eval_libero.py`` exactly:

  * views  : agentview + eye-in-hand, rotated 180 deg to match training preprocessing
             (done by ``sim_libero.get_libero_image``), optionally resized to IMAGE_SIZE.
  * state  : 8-d = [eef_pos(3), quat2axisangle(eef_quat)(3), gripper_qpos(2)].
  * action : predict_action -> normalized [chunk, 7]; min/max un-normalized; gripper
             binarized to {0,1} (1=open) then remapped to LIBERO OSC_POSE {-1 open, +1 close}
             via 1-2*g (the same convention fix validated in the open-loop study).

Selected at runtime by the sim harness via:
    POLICY_FACTORY=vlajepa_libero_policy:build_policy

Env: MODEL_REPO, CKPT_REL, BASE_VLM, BASE_ENCODER, DTYPE, UNNORM_KEY, HF_TOKEN,
     REPLAN_STEPS, NUM_STEPS_WAIT, IMAGE_SIZE (0 = native render res, no resize).
Requires /ryzers (model_smoke helpers), /repos/VLA-JEPA and /opt/sim on PYTHONPATH
(the demo scripts set this).
"""
import math
import os

import numpy as np
import torch
from PIL import Image

from sim_libero.libero_env import get_libero_image
from sim_libero.policy import Policy

# Reuse the validated checkpoint resolver + config repointer shipped with the package.
from model_smoke import repoint_config, resolve_checkpoint

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def _quat2axisangle(quat):
    """robosuite xyzw quaternion -> 3-d axis-angle (verbatim from upstream eval_libero.py)."""
    q = np.asarray(quat, dtype=np.float32).copy()
    q[3] = min(1.0, max(-1.0, float(q[3])))
    den = math.sqrt(1.0 - q[3] * q[3])
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    return (q[:3] * 2.0 * math.acos(q[3]) / den).astype(np.float32)


def _unnormalize(normalized, stats):
    """Inverse min/max normalization + gripper binarize to {0,1} (open=1). Mirrors upstream."""
    lo = np.asarray(stats["min"], dtype=np.float32)
    hi = np.asarray(stats["max"], dtype=np.float32)
    mask = np.asarray(stats.get("mask", np.ones_like(lo, dtype=bool)))
    a = np.clip(np.asarray(normalized, dtype=np.float32), -1.0, 1.0)
    if a.shape[-1] >= 7:
        a[:, 6] = np.where(a[:, 6] < 0.5, 0.0, 1.0)
    return np.where(mask, 0.5 * (a + 1.0) * (hi - lo) + lo, a)


class VJepaLiberoPolicy(Policy):
    name = "vla-jepa"

    def __init__(self, model, action_stats, chunk, replan_steps, num_steps_wait, image_size):
        self.model = model
        self.action_stats = action_stats
        self.chunk = int(chunk)
        self.replan_steps = int(replan_steps)
        self.num_steps_wait = int(num_steps_wait)
        self.image_size = int(image_size) if image_size else 0

    def reset(self, instruction):
        # VLA-JEPA inference is stateless: the V-JEPA2 world model is a train-time
        # regularizer and is not called by predict_action. Nothing to cache per episode.
        pass

    def _to_pil(self, arr):
        img = Image.fromarray(np.ascontiguousarray(arr))
        if self.image_size and img.size != (self.image_size, self.image_size):
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        return img

    @torch.no_grad()
    def predict_action_chunk(self, obs, instruction):
        views = get_libero_image(obs)  # agentview + wrist, rotated 180 deg (training orient)
        imgs = [self._to_pil(views["image"]), self._to_pil(views["wrist_image"])]
        state = np.concatenate([
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32).reshape(-1),
            _quat2axisangle(obs["robot0_eef_quat"]),
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1),
        ]).astype(np.float32)[None]  # (1, 8)

        out = self.model.predict_action(
            batch_images=[imgs], instructions=[instruction], state=[state])
        raw = _unnormalize(np.asarray(out["normalized_actions"], dtype=np.float32)[0],
                           self.action_stats)  # (chunk, 7)
        # Gripper: model emits {0,1} (1=open); LIBERO OSC_POSE wants {-1 open, +1 close}.
        if raw.shape[-1] >= 7:
            raw[:, 6] = 1.0 - 2.0 * raw[:, 6]
        return np.asarray(raw, dtype=np.float32)


def build_policy():
    # ryzers passes optional knobs as empty strings; treat "" as unset.
    dtype = _DTYPES[os.environ.get("DTYPE") or "bfloat16"]
    ckpt = resolve_checkpoint()
    repoint_config(ckpt)

    # Warm the base models into the HF cache (predict path constructs them on load).
    from huggingface_hub import snapshot_download
    for repo in (os.environ.get("BASE_VLM") or "Qwen/Qwen3-VL-2B-Instruct",
                 os.environ.get("BASE_ENCODER") or "facebook/vjepa2-vitl-fpc64-256"):
        snapshot_download(repo_id=repo, token=os.environ.get("HF_TOKEN") or None)

    from starVLA.model.framework.base_framework import baseframework
    from starVLA.model.tools import read_mode_config

    _, norm_stats = read_mode_config(ckpt)
    unnorm_key = os.environ.get("UNNORM_KEY") or next(iter(norm_stats.keys()))
    action_stats = norm_stats[unnorm_key]["action"]

    model = baseframework.from_pretrained(ckpt).to("cuda:0").to(dtype).eval()
    chunk = model.config.framework.action_model.future_action_window_size + 1
    # Upstream eval replans once per full chunk (step % chunk == 0); default to that.
    replan = int(os.environ.get("REPLAN_STEPS") or chunk)
    wait = int(os.environ.get("NUM_STEPS_WAIT") or 10)  # upstream LIBERO settle count
    image_size = int(os.environ.get("IMAGE_SIZE") or 0)  # 0 -> native render res (open-loop path)

    print(f"[vlajepa_libero_policy] ready (ckpt={ckpt}, unnorm_key={unnorm_key}, "
          f"chunk={chunk}, replan={replan}, wait={wait}, "
          f"image_size={image_size or 'native'})", flush=True)
    return VJepaLiberoPolicy(model, action_stats, chunk, replan, wait, image_size)
