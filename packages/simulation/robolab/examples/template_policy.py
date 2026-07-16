# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""COPY-ME template: plug your own VLA/WAM into the RoboLab-AMD simulator base.

The simulation/robolab image is model-agnostic. To drive it with your model you do NOT edit
this package: you build YOUR ryzer FROM the sim base and ship an adapter like this one, then
select it at runtime via POLICY_FACTORY.

Three steps
-----------
1. In your model package's Dockerfile, chain on the sim base:

       ARG BASE_IMAGE
       FROM ${BASE_IMAGE}
       # install your model UNDER the base's torch+numpy pins (robosuite needs numpy 1.26.4).
       COPY adapters/ /opt/model-adapters/

   Build the chain:   ryzers build simulation/robolab <yourmodel> --name <yourmodel>-robolab

2. Implement the two methods below (load the model in build_policy, convert obs -> joint-space
   action chunk in predict_action_chunk).

3. Run, pointing POLICY_FACTORY at your factory and putting the adapter on PYTHONPATH:

       PYTHONPATH=/opt/model-adapters:$PYTHONPATH \
       POLICY_FACTORY=template_policy:build_policy \
         ryzers run --name <yourmodel>-robolab /ryzers/demos/demo_sim_sanity.sh

Contract
--------
obs           : dict from the harness (sim_robolab.scene.Scene.observe):
                  obs["video"]      [V, H, W, 3] float32 in [-1, 1] -- the 3 cameras
                                    (robot0_eye_in_hand, agentview, sideview)
                  obs["view"]       [H, V*W, 3] uint8 stitched viewport (display only)
                  obs["proprios"]   [16]  eef xyz(3) + quat wxyz(4) + gripper(1) + 8 zeros
                  obs["qpos"]       [7]   arm joint angles (rad) -- anchor for joint targets
                  obs["instruction"] str, the natural-language task
instruction   : str, the natural-language task.
return        : np.ndarray [T, 8] float32 = (q0..q6, gripper),
                7 ABSOLUTE Franka joint targets (rad) + gripper in [-1, 1] (-1 open, +1 close).
                The harness maps absolute targets -> the JointPositionController's per-step
                normalised delta, executes the first `replan_steps` rows, then calls you again.
                (This matches Cosmos3-Nano-Policy-DROID's joint-position action head.)
"""
import numpy as np

from sim_robolab.policy import Policy


class TemplatePolicy(Policy):
    name = "template"
    replan_steps = 16     # env steps executed per predicted chunk before replanning
    num_steps_wait = 0

    def __init__(self, model):
        self.model = model

    def reset(self, instruction):
        # Called once per episode. Clear any per-episode caches / receding-horizon queue.
        pass

    def predict_action_chunk(self, obs, instruction):
        # TODO: preprocess obs into your model's inputs (use obs["video"] + obs["proprios"]/
        # obs["qpos"]), run inference, and return ABSOLUTE joint targets + gripper as above.
        raise NotImplementedError("wire your model here")
        # return np.tile(np.append(obs["qpos"], -1.0), (self.replan_steps, 1)).astype(np.float32)


def build_policy():
    # TODO: load your checkpoint / processor once here (env knobs: CKPT, DATASET_STATS, ...).
    model = None
    return TemplatePolicy(model)
