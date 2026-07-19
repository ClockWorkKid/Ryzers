# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""COPY-ME: annotated skeleton for plugging a model into the MimicGen simulator harness.

Two integration styles, both selected at runtime via POLICY_FACTORY=module:function:

  (A) In-process policy. Your model runs in the harness venv (numpy 1.26 / robosuite 1.4). Subclass
      sim_mimicgen.Policy and implement per-step predict_action. Simplest when your model's deps
      are compatible with the sim base.

  (B) Remote policy server (recommended for heavy / dep-incompatible models — this is how VERA plugs
      in). Run your model as a websocket policy server (obs->action-chunk) in its own venv/image,
      and reuse the shipped sim_mimicgen.remote_policy:build_policy as the factory — no adapter code
      needed, just POLICY_HOST/POLICY_PORT. Only write an adapter (below) if your server speaks a
      different protocol.

Then:
    POLICY_FACTORY=template_policy:build_policy ryzers run --name <yourmodel>-mimicgen \\
      TASK=stack_d0 /ryzers/demos/demo_closedloop.sh
"""
import numpy as np

from sim_mimicgen.policy import Policy, PolicyObservation, PolicyOutput


class TemplatePolicy(Policy):
    """(A) In-process example. Replace predict_action with your model's per-step inference."""

    name = "template"
    cfg = None
    device = None

    # Runner hints the harness reads (or leave None for the 2-view / ctx-9 defaults).
    view_keys = ["agentview_image", "robot0_eye_in_hand_image"]
    context_frames = 9
    render_size = 128

    def __init__(self):
        # TODO: load your model / weights here.
        self._action_dim = 7  # robosuite OSC_POSE: [dx,dy,dz, drx,dry,drz, gripper]

    def reset(self) -> None:
        """Called once per episode (clear rolling caches, action queue, etc.)."""

    def warmup_obs(self, obs: PolicyObservation) -> None:
        """Optional: consume context frames during demo warmup without inferring."""

    def predict_action(self, obs: PolicyObservation) -> PolicyOutput:
        # obs.rgb is the width-concatenated multi-view frame (H, W*len(view_keys), 3).
        # TODO: run your model and return one env-step action (shape (1, action_dim)).
        action = np.zeros((1, self._action_dim), dtype=np.float32)
        return PolicyOutput(action=action, info=None)


def build_policy() -> Policy:
    return TemplatePolicy()
