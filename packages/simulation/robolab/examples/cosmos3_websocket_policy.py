# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""RoboLab-AMD client adapter for the Cosmos3-Nano-Policy-DROID policy server.

This is the G4 seam: it turns the harness observation into the DROID observation dict the
NVIDIA cosmos-framework `action_policy_server_robolab` expects, ships it over OpenPI's
msgpack+WebSocket protocol, and returns the server's `[T, 8]` joint-position action chunk in
our harness convention (7 absolute joint targets + gripper in [-1, 1]).

The policy model itself (16B Cosmos3 MoT) runs in the separate `cosmos3` image as a policy
server; this adapter only speaks its wire protocol, so the sim image stays model-free.

Wiring (server already running on :8000):

    PYTHONPATH=/opt/robolab/vendor:/opt/robolab/examples:$PYTHONPATH \
    POLICY_FACTORY=cosmos3_websocket_policy:build_policy \
    POLICY_SERVER_HOST=127.0.0.1 POLICY_SERVER_PORT=8000 \
      python -m sim_robolab.closedloop

Observation contract the server wants (see action_policy_server_robolab.py):
    prompt                                  str
    observation/wrist_image_left            [H,W,3] uint8   (our video[0], robot0_eye_in_hand)
    observation/exterior_image_1_left       [H,W,3] uint8   (our video[1], agentview stand-in)
    observation/exterior_image_2_left       [H,W,3] uint8   (our video[2], sideview stand-in)
    observation/joint_position              [7]  float32    (our qpos)
    observation/gripper_position            [1]  float32    (DROID: 0=open .. 1=closed)
Server returns {"action": [T, 8]} = 7 abs joint targets + gripper in the SAME gripper space.

Gripper convention: our sim openness is 0=closed..1=open; DROID gripper_position is
0=open..1=closed, so we send `1 - openness` and map the returned gripper g (0=open..1=closed)
to our command via `2g - 1` (-1 open, +1 close). If a run shows the gripper acting inverted,
set GRIPPER_INVERT=1 to flip both directions together.
"""
import os

import numpy as np

from sim_robolab.policy import Policy


def _to_uint8(view_norm):
    """[-1, 1] float32 HxWx3 -> uint8 HxWx3."""
    arr = (np.asarray(view_norm, dtype=np.float32) + 1.0) * 127.5
    return np.clip(arr, 0, 255).astype(np.uint8)


class Cosmos3WebsocketPolicy(Policy):
    name = "cosmos3-droid"

    def __init__(self, host, port, replan_steps=8, prompt_override=None, gripper_invert=False):
        from openpi_client.websocket_client_policy import WebsocketClientPolicy

        self.replan_steps = int(replan_steps)
        self.num_steps_wait = 0
        self._prompt_override = prompt_override
        self._gripper_invert = bool(gripper_invert)
        self._debug = os.environ.get("COSMOS_DEBUG", "0") not in ("0", "", "false", "False")
        print(f"[cosmos3] connecting to policy server ws://{host}:{port} ...", flush=True)
        self._client = WebsocketClientPolicy(host=host, port=int(port))
        md = self._client.get_server_metadata()
        print(f"[cosmos3] connected; server metadata={md}", flush=True)
        self.last_infer_ms = None
        self.infer_ms_log = []

    def reset(self, instruction):
        self._client.reset()

    def predict_action_chunk(self, obs, instruction):
        prompt = self._prompt_override or instruction
        video = obs["video"]  # [3, H, W, 3] float32 in [-1, 1]: [wrist, exterior1, exterior2]
        openness = float(np.asarray(obs["proprios"], dtype=np.float32)[7])  # 0 closed .. 1 open
        g_cli = openness if self._gripper_invert else (1.0 - openness)  # DROID: 0 open .. 1 closed

        request = {
            "prompt": prompt,
            "observation/wrist_image_left": _to_uint8(video[0]),
            "observation/exterior_image_1_left": _to_uint8(video[1]),
            "observation/exterior_image_2_left": _to_uint8(video[2]),
            "observation/joint_position": np.asarray(obs["qpos"], dtype=np.float32).reshape(7),
            "observation/gripper_position": np.asarray([g_cli], dtype=np.float32),
        }

        result = self._client.infer(request)
        action = np.asarray(result["action"], dtype=np.float32)  # [T, 8]
        if action.ndim != 2 or action.shape[1] < 8:
            raise ValueError(f"unexpected server action shape {action.shape}, expected [T, 8]")

        g_out = action[:, 7]  # DROID space (0 open .. 1 closed) unless inverted

        if self._debug:
            cur_q = np.asarray(obs["qpos"], dtype=np.float32).reshape(7)
            jt = action[:, :7]
            djoint = jt - cur_q[None, :]  # per-row target-minus-current (abs vs delta tell)
            print(
                f"[cosmos3.dbg] chunk T={action.shape[0]} "
                f"joint_tgt[min={jt.min():+.3f} max={jt.max():+.3f}] "
                f"cur_q[min={cur_q.min():+.3f} max={cur_q.max():+.3f}] "
                f"|tgt-cur|[max={np.abs(djoint).max():.3f} mean={np.abs(djoint).mean():.3f}] "
                f"grip_raw[min={g_out.min():.3f} max={g_out.max():.3f} last={g_out[-1]:.3f}] "
                f"in_openness={openness:.3f} g_cli={g_cli:.3f}",
                flush=True,
            )

        cmd = (1.0 - 2.0 * g_out) if self._gripper_invert else (2.0 * g_out - 1.0)  # -> [-1, 1]
        action = action.copy()
        action[:, 7] = np.clip(cmd, -1.0, 1.0)

        timing = result.get("server_timing") or {}
        self.last_infer_ms = timing.get("infer_ms")
        if self.last_infer_ms is not None:
            self.infer_ms_log.append(float(self.last_infer_ms))
        return action.astype(np.float32)


def build_policy():
    host = os.environ.get("POLICY_SERVER_HOST", "127.0.0.1")
    port = os.environ.get("POLICY_SERVER_PORT", "8000")
    replan = os.environ.get("REPLAN_STEPS", "8")
    prompt = os.environ.get("PROMPT") or None
    invert = os.environ.get("GRIPPER_INVERT", "0") not in ("0", "", "false", "False")
    return Cosmos3WebsocketPolicy(
        host=host, port=port, replan_steps=replan, prompt_override=prompt, gripper_invert=invert
    )
