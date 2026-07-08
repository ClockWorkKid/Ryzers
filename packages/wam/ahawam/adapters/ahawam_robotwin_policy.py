# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""AHA-WAM RoboTwin policy adapter for the simulation/robotwin interactive harness.

Implements the model-agnostic `sim_robotwin.Policy` seam by wrapping the AHA-WAM two-phase
world-action model through the *validated* RoboTwin deploy policy from
experiments/robotwin/ahawam_policy/deploy_policy.py (WorldActionRobotWinPolicy: config
compose, model instantiate + checkpoint load, processor/normalizer, the camera-composited
image + state builders, and the async video-prefill + action-chunk schedule driven by
chunks_per_video_prefill). Reusing that plugin verbatim means interactive rollouts match the
closed-loop numbers.

Selected at runtime by the sim harness via
  POLICY_FACTORY=ahawam_robotwin_policy:build_policy

The parity-critical closed-loop path does NOT use this adapter: it runs RoboTwin's own
script/eval_policy.py against the same deploy_policy.py plugin (experiments/robotwin/
eval_robotwin_single.py). This wrapper only bridges that plugin's per-chunk inference into
the interactive Policy seam (both interactive_server and interactive_server_rt).

Env: CKPT, DATASET_STATS, MIXED_PRECISION (bf16), NUM_INFERENCE_STEPS, ACTION_HORIZON,
CHUNKS_PER_VIDEO_PREFILL. Requires /repos/ahawam, /repos/ahawam/experiments/robotwin and
/opt/sim on PYTHONPATH (the demo sets this).
"""
import os

import numpy as np

import experiments.robotwin.ahawam_policy.deploy_policy as D
from sim_robotwin.policy import Policy

DEFAULT_CKPT = "/models/ahawam_release/robotwin_ahawam-flash.pt"
DEFAULT_STATS = "/models/ahawam_release/dataset_stats.json"


class AhawamRoboTwinPolicy(Policy):
    name = "ahawam"

    def __init__(self, model):
        self.model = model  # deploy_policy.WorldActionRobotWinPolicy
        # One predicted chunk == action_chunk_size sim steps; replan every chunk so the
        # harness feeds a fresh observation each cycle (matches the deploy queue cadence).
        self.replan_steps = int(self.model.model.action_chunk_size)

    def reset(self, instruction):
        self.model.reset()

    def predict_action_chunk(self, obs, instruction):
        m = self.model
        # Mirror WorldActionRobotWinPolicy._fill_action_queue's soft-reset cadence: begin a
        # new video prefill once the current prefill's chunk budget is exhausted, otherwise
        # roll the next action chunk against the cached video KV state.
        state = getattr(m.model, "_inference_state", None)
        next_chunk_index = 0 if state is None else int(state.get("next_chunk_index", 0))
        if (
            next_chunk_index >= m.chunks_per_video_prefill
            or m._chunks_since_video_prefill >= m.chunks_per_video_prefill
        ):
            m._soft_reset_for_new_observation()
        chunk = m._predict_next_chunk(observation=obs, instruction=instruction)
        m._chunks_since_video_prefill += 1
        return np.asarray(chunk, dtype=np.float32)


def build_policy():
    # ryzers passes optional knobs as empty strings; treat "" as unset.
    usr_args = {
        "ckpt_setting": os.environ.get("CKPT") or DEFAULT_CKPT,
        "dataset_stats_path": os.environ.get("DATASET_STATS") or DEFAULT_STATS,
        "device": "cuda",  # ROCm torch reports as cuda; deploy_policy falls back to cpu if absent
        "mixed_precision": os.environ.get("MIXED_PRECISION") or "bf16",
    }
    for key, env in (("num_inference_steps", "NUM_INFERENCE_STEPS"),
                     ("action_horizon", "ACTION_HORIZON"),
                     ("chunks_per_video_prefill", "CHUNKS_PER_VIDEO_PREFILL")):
        val = os.environ.get(env)
        if val:
            usr_args[key] = val

    model = D.get_model(usr_args)
    print(
        f"[ahawam_robotwin_policy] model ready (ckpt={usr_args['ckpt_setting']}, "
        f"chunk={model.model.action_chunk_size}, "
        f"chunks_per_video_prefill={model.chunks_per_video_prefill}, "
        f"num_inference_steps={model.num_inference_steps})",
        flush=True,
    )
    return AhawamRoboTwinPolicy(model)
