# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""VERA adapter for the model-agnostic MimicGen simulator harness (sim_mimicgen).

VERA serves its WAN video planner + Jacobian IDM over the websocket obs->action-chunk protocol
(vera.server.start_server_mimicgen). That protocol is exactly what the sim base's
`sim_mimicgen.remote_policy.RemoteWebsocketPolicy` speaks, so VERA needs no bespoke translation —
this factory just wraps it with VERA-friendly defaults (per-rollout WAN text prompt, host/port
from env) and is selected by the harness via:

    POLICY_FACTORY=vera_mimicgen_policy:build_policy

demos/demo_closedloop_mimicgen.sh starts the VERA policy server, then runs the sim base harness
with this factory pointed at it (POLICY_PORT). The env stepping, reset-to-demo, success tracking
and video all live in the sim base; VERA only provides the policy behind the socket.
"""
import os

from sim_mimicgen.policy import Policy
from sim_mimicgen.remote_policy import RemoteWebsocketPolicy


def build_policy() -> Policy:
    return RemoteWebsocketPolicy(
        host=os.environ.get("POLICY_HOST", "127.0.0.1"),
        port=int(os.environ.get("POLICY_PORT", os.environ.get("PORT", 8800))),
        context_frames=(int(os.environ["CONTEXT_FRAMES"]) if os.environ.get("CONTEXT_FRAMES") else None),
        render_size=(int(os.environ["RENDER_SIZE"]) if os.environ.get("RENDER_SIZE") else None),
        prompt=os.environ.get("PROMPT") or None,
    )
