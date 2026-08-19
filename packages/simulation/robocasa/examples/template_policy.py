# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""COPY-ME template: plug your own VLA/WAM into the RoboCasa simulator base.

The simulation/robocasa image is model-agnostic. To drive it with your model you do NOT
edit this package: you build YOUR ryzer FROM the sim base and ship an adapter like this
one, then select it at runtime.

Three steps
-----------
1. In your model package's Dockerfile, chain on the sim base:

       ARG BASE_IMAGE
       FROM ${BASE_IMAGE}
       # install your model UNDER the base's torch+numpy pins so you don't break the
       # RoboCasa/robosuite/MuJoCo stack (it needs numpy 1.26.4).
       COPY adapters/ /opt/model-adapters/

   Build the chain:   ryzers build simulation/robocasa <yourmodel> --name <yourmodel>-robocasa

2. Implement the two methods below (load the model in build_policy, convert obs -> action
   chunk in predict_action_chunk).

3. Run, pointing POLICY_FACTORY at your factory and putting the adapter on PYTHONPATH:

       PYTHONPATH=/opt/model-adapters:$PYTHONPATH \
       POLICY_FACTORY=template_policy:build_policy \
         ryzers run --name <yourmodel>-robocasa /ryzers/demos/demo_interactive.sh

The harness owns the env, rendering, MJPEG streaming and the episode loop. Your adapter
only turns one (obs, instruction) into an action chunk.

Contract
--------
obs           : dict from the harness (sim_robocasa.scene.Scene.observe):
                  obs["video"]      [V, H, W, 3] float32 in [-1, 1] -- the 3 cameras
                                    (agentview_left, agentview_right, eye_in_hand)
                  obs["view"]       [H, V*W, 3] uint8 stitched viewport (display only)
                  obs["proprios"]   [16]  eef xyz(3) + quat wxyz(4) + gripper(1) + 8 zeros
                  obs["instruction"] str, the natural-language task
instruction   : str, the natural-language task.
return        : np.ndarray [T, 7] float32 = (dx, dy, dz, drx, dry, drz, gripper),
                robosuite OSC_POSE delta control. The harness zero-pads each row into the
                robot's full action_spec, executes the first `replan_steps` rows, then
                calls you again.
"""
import numpy as np

from sim_robocasa.policy import Policy


class TemplatePolicy(Policy):
    name = "template"
    replan_steps = 32     # env steps executed per predicted chunk before replanning
    num_steps_wait = 0    # RoboCasa needs no settle steps

    def __init__(self, model):
        self.model = model

    def reset(self, instruction):
        # Called once per episode. Clear any per-episode caches / receding-horizon queue.
        pass

    def predict_action_chunk(self, obs, instruction):
        # TODO: preprocess obs into your model's inputs (use obs["video"] + obs["proprios"]),
        # run inference, and return actions in the robosuite OSC_POSE delta space above.
        raise NotImplementedError("wire your model here")
        # return np.zeros((self.replan_steps, 7), dtype=np.float32)


def build_policy():
    # TODO: load your checkpoint / processor once here (env knobs: CKPT, DATASET_STATS, ...).
    model = None
    return TemplatePolicy(model)
