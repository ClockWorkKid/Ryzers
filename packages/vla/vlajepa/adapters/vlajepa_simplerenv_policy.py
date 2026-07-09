# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""VLA-JEPA SimplerEnv (WidowX / BridgeData v2) policy adapter for the sim_simplerenv harness.

Implements the model-agnostic ``sim_simplerenv.Policy`` seam by wrapping VLA-JEPA (the starVLA
``baseframework``). Mirrors the upstream SimplerEnv eval interface verbatim
(``examples/SimplerEnv/eval_files/model2simpler_interface.py``) so real2sim rollouts match the
shipped paper setup:

  * view    : SINGLE third-person camera (the WidowX/Bridge digital twin exposes only
              ``3rd_view_camera``; there is no wrist cam), resized to IMAGE_SIZE (224) with
              INTER_AREA. No 180-deg rotation (unlike LIBERO).
  * state   : NOT passed. The bridge inference path is image + instruction only (state_dim 8 in
              the checkpoint config is the train-time GR00T proprio layout; upstream's websocket
              step() sends no state). We fall back to a zero state only if the loaded
              ``predict_action`` requires the kwarg.
  * action  : predict_action -> normalized [chunk, 7]; q01/q99 un-normalized; gripper (idx 6)
              binarized to {0,1} (1=open). Rows are temporally ensembled every step with the
              AdaptiveEnsembler (widowx horizon 7, alpha 0.1) -> one action/step, then emitted as
              [dx, dy, dz, rot_axangle(3), gripper] for the ManiSkill3 delta-ee control mode:
                world_vector = raw[:3]; rot_axangle = euler2axangle(raw[3:6]);
                gripper      = GRIPPER_OPEN_SIGN * (2*(g>0.5) - 1).

Cadence: because upstream re-predicts and temporally ensembles at EVERY env step, this adapter
sets ``replan_steps = 1`` and does the ensembling internally, returning a single-row [1, 7] chunk.

The ManiSkill3 WidowX gripper sign is validated empirically: upstream ManiSkill2 used
+1=open for widowx_bridge, but the ManiSkill3 real2sim port may invert it. ``GRIPPER_OPEN_SIGN``
(default +1) makes it a one-flag flip if a smoke rollout shows grasps failing.

Selected at runtime by the sim harness via:
    POLICY_FACTORY=vlajepa_simplerenv_policy:build_policy

Env: MODEL_REPO, CKPT_REL (default SimplerEnv/checkpoints/VLA-JEPA-SimplerEnv.pt), BASE_VLM,
     BASE_ENCODER, DTYPE, UNNORM_KEY (default oxe_bridge), HF_TOKEN, IMAGE_SIZE (default 224),
     ENSEMBLE_ALPHA, ENSEMBLE_HORIZON, USE_DDIM, NUM_DDIM_STEPS, GRIPPER_OPEN_SIGN.
Requires /ryzers (model_smoke helpers), /repos/VLA-JEPA and /opt/sim on PYTHONPATH.
"""
import os

import numpy as np
import torch

from sim_simplerenv.policy import ACTION_DIM, Policy
from sim_simplerenv import simplerenv_env as se

# Reuse the validated checkpoint resolver + config repointer shipped with the package.
from model_smoke import repoint_config, resolve_checkpoint

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def _resize(img, size):
    """Resize HxWx3 uint8 to (size, size) with INTER_AREA (matches upstream _resize_image)."""
    import cv2 as cv

    if size and (img.shape[0], img.shape[1]) != (size, size):
        img = cv.resize(img, (size, size), interpolation=cv.INTER_AREA)
    return np.ascontiguousarray(img.astype(np.uint8))


def _unnormalize(normalized, stats):
    """Inverse q01/q99 normalization + gripper binarize to {0,1} (open=1). Mirrors upstream."""
    lo = np.asarray(stats["q01"], dtype=np.float32)
    hi = np.asarray(stats["q99"], dtype=np.float32)
    mask = np.asarray(stats.get("mask", np.ones_like(lo, dtype=bool)))
    a = np.clip(np.asarray(normalized, dtype=np.float32), -1.0, 1.0)
    if a.shape[-1] >= 7:
        a[:, 6] = np.where(a[:, 6] < 0.5, 0.0, 1.0)
    return np.where(mask, 0.5 * (a + 1.0) * (hi - lo) + lo, a)


class VJepaSimplerPolicy(Policy):
    name = "vla-jepa"
    replan_steps = 1     # upstream re-predicts + temporally ensembles every env step
    num_steps_wait = 0   # ManiSkill3 real2sim resets to a stable pose; no settle needed

    def __init__(self, model, action_stats, image_size, ensembler, open_sign,
                 use_ddim, num_ddim_steps, replan_steps=1):
        self.model = model
        self.action_stats = action_stats
        self.image_size = int(image_size) if image_size else 0
        self.ensembler = ensembler
        self.open_sign = float(open_sign)
        self.use_ddim = use_ddim
        self.num_ddim_steps = int(num_ddim_steps)
        # replan_steps=1 (default) => per-step temporal ensembling (faithful eval cadence).
        # replan_steps>1 => chunk-replay: emit REPLAN_STEPS rows, ensembling disabled (smoother
        # interactive demos). Set via REPLAN_STEPS in build_policy.
        self.replan_steps = max(1, int(replan_steps))
        self.task_description = None
        self._needs_state = None  # lazily detected: does predict_action require a state kwarg?

    def reset(self, instruction):
        self.task_description = instruction
        if self.ensembler is not None:
            self.ensembler.reset()

    def _image(self, obs):
        """Single third-person RGB (HxWx3 uint8), resized. Reads obs alone (env not needed)."""
        sensor = obs["sensor_data"] if isinstance(obs, dict) and "sensor_data" in obs else {}
        cam = ("3rd_view_camera" if "3rd_view_camera" in sensor
               else "overhead_camera" if "overhead_camera" in sensor
               else (sorted(sensor)[0] if sensor else None))
        img = sensor[cam]["rgb"]
        if hasattr(img, "cpu"):
            img = img.cpu().numpy()
        img = np.asarray(img)
        if img.ndim == 4:
            img = img[0]
        return _resize(img, self.image_size)

    def _predict(self, img, instruction):
        """Call predict_action image+instruction-only; fall back to a zero state if required."""
        kw = dict(batch_images=[[img]], instructions=[instruction])
        if self._needs_state is None:
            try:
                out = self.model.predict_action(**kw)
                self._needs_state = False
                return out
            except TypeError:
                self._needs_state = True
        if self._needs_state:
            kw["state"] = [np.zeros((1, 8), dtype=np.float32)]
        return self.model.predict_action(**kw)

    def _to_env_action(self, row):
        """One raw [x,y,z,roll,pitch,yaw,gripper] row -> env [dx,dy,dz,rot_axangle(3),gripper]."""
        from transforms3d.euler import euler2axangle

        row = np.asarray(row, dtype=np.float32).reshape(-1)
        axes, angle = euler2axangle(*(float(v) for v in row[3:6]))
        rot_axangle = np.asarray(axes, dtype=np.float32) * float(angle)
        gripper = self.open_sign * (2.0 * float(row[6] > 0.5) - 1.0)
        return np.concatenate([row[:3], rot_axangle, [gripper]]).astype(np.float32)

    @torch.no_grad()
    def predict_action_chunk(self, obs, instruction):
        img = self._image(obs)
        out = self._predict(img, instruction or self.task_description or "")
        norm = np.asarray(out["normalized_actions"], dtype=np.float32)[0]  # (chunk, 7)
        raw = _unnormalize(norm, self.action_stats)                        # (chunk, 7)

        if self.replan_steps > 1:
            # Chunk-replay (interactive demos): emit multiple rows, no temporal ensembling.
            rows = raw[: self.replan_steps]
            return np.stack([self._to_env_action(r) for r in rows]).astype(np.float32)

        # Default eval cadence: temporally ensemble every step -> a single [1, 7] row.
        ens = self.ensembler.ensemble_action(raw) if self.ensembler is not None else raw[0]
        return self._to_env_action(ens)[None]  # (1, 7); harness re-predicts each step


def build_policy():
    # ryzers passes optional knobs as empty strings; treat "" as unset.
    # NOTE: CKPT_REL must be set in the environment (demo_closedloop_simplerenv.sh) BEFORE this
    # module is imported -- model_smoke binds its CKPT_REL global at import time.
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
    unnorm_key = os.environ.get("UNNORM_KEY") or ("oxe_bridge" if "oxe_bridge" in norm_stats
                                                  else next(iter(norm_stats.keys())))
    action_stats = norm_stats[unnorm_key]["action"]

    model = baseframework.from_pretrained(ckpt).to("cuda:0").to(dtype).eval()
    image_size = int(os.environ.get("IMAGE_SIZE") or 224)
    open_sign = float(os.environ.get("GRIPPER_OPEN_SIGN") or 1.0)
    use_ddim = (os.environ.get("USE_DDIM", "true").lower() not in ("0", "false", "no"))
    num_ddim = int(os.environ.get("NUM_DDIM_STEPS") or 10)

    # AdaptiveEnsembler (temporal ensembling) from the upstream SimplerEnv eval files.
    # widowx_bridge uses horizon 7 / alpha 0.1 (see upstream model2simpler_interface + paper appendix).
    ensembler = None
    try:
        from examples.SimplerEnv.eval_files.adaptive_ensemble import AdaptiveEnsembler
        horizon = int(os.environ.get("ENSEMBLE_HORIZON") or 7)
        alpha = float(os.environ.get("ENSEMBLE_ALPHA") or 0.1)
        ensembler = AdaptiveEnsembler(horizon, alpha)
    except Exception as e:  # noqa: BLE001 - fall back to raw first-row action if unavailable
        print(f"[vlajepa_simplerenv_policy] AdaptiveEnsembler unavailable ({e}); "
              "using raw first-row action (no temporal ensemble)", flush=True)

    replan_steps = max(1, int(os.environ.get("REPLAN_STEPS") or 1))
    if replan_steps > 1:  # chunk-replay mode: ensembling is meaningless, disable it
        ensembler = None

    print(f"[vlajepa_simplerenv_policy] ready (ckpt={ckpt}, unnorm_key={unnorm_key}, "
          f"image_size={image_size}, replan_steps={replan_steps}, "
          f"ensemble={'on' if ensembler is not None else 'off'}, "
          f"open_sign={open_sign:+.0f}, use_ddim={use_ddim}, num_ddim={num_ddim})", flush=True)
    return VJepaSimplerPolicy(model, action_stats, image_size, ensembler, open_sign,
                              use_ddim, num_ddim, replan_steps)
