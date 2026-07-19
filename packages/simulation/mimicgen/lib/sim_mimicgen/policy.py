# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Model-agnostic policy seam for the MimicGen simulator harness.

The harness owns the env, rendering, reset-to-demo, success tracking and the episode loop
(via upstream `vera.env_runner.MimicgenRunner`, reused unchanged per rule 2.1). A policy only
turns per-step observations into actions. Any model (VERA, ...) ships a factory
`build_policy() -> Policy` and is selected at runtime with `POLICY_FACTORY=module:function`
(default: the built-in `RandomPolicy`, which needs no model / no server).

Seam contract. MimicGen tasks run robosuite's OSC_POSE controller, so the runner drives the
policy **per env step** through the upstream `BasePolicy` interface:

    predict_action(obs: PolicyObservation) -> PolicyOutput   # obs.rgb is the (view-concat) frame
    reset() -> None                                          # once per episode

Chunked / remote models (e.g. VERA's WAN planner + Jacobian IDM served over websocket) bridge
"predict a chunk, replay one step at a time" internally — see `remote_policy.RemoteWebsocketPolicy`,
which maintains a rolling context window + an action queue and only calls the model on refill.

Policies may advertise runner hints as attributes the harness reads when building the env cfg:

    view_keys:        list[str]  # image obs keys concatenated width-wise into obs.rgb
    context_frames:   int        # frames warmed up before the policy first acts
    render_size:      int        # per-view render edge (px)
"""
import importlib
import os

from vera.policy.base_policy import BasePolicy, PolicyObservation, PolicyOutput  # noqa: F401


class Policy(BasePolicy):
    """Model-agnostic seam (per-step `predict_action` / `reset`, from upstream `BasePolicy`)."""

    name = "policy"

    # Optional runner hints (subclasses / factories may override). None => harness default.
    view_keys = None        # e.g. ["agentview_image", "robot0_eye_in_hand_image"]
    context_frames = None   # e.g. 9 (1 + (N-1)*stride); warmup = context_frames - 1
    render_size = None       # e.g. 128

    def warmup_obs(self, obs: PolicyObservation) -> None:
        """Optional: fill context without inferring (called during demo-state warmup)."""


def load_policy() -> Policy:
    """Instantiate the policy named by POLICY_FACTORY=module:function (default RandomPolicy)."""
    spec = os.environ.get("POLICY_FACTORY") or "sim_mimicgen.random_policy:build_policy"
    if ":" not in spec:
        raise ValueError(f"POLICY_FACTORY must be 'module:function', got {spec!r}")
    module_name, fn_name = spec.split(":", 1)
    factory = getattr(importlib.import_module(module_name), fn_name)
    policy = factory()
    if not isinstance(policy, BasePolicy):
        raise TypeError(f"{spec} did not return a sim_mimicgen.Policy (got {type(policy)})")
    return policy


__all__ = ["Policy", "load_policy", "PolicyObservation", "PolicyOutput"]
