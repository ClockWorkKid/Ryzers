# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""MolmoAct2 LIBERO policy adapter for the shared simulation/libero harness (bridge client).

Implements the model-agnostic `sim_libero.Policy` seam. MolmoAct2's LIBERO policy is the
allenai lerobot stack (lerobot 0.5.1 hard-pins numpy>=2, transformers 5.3), which cannot
co-exist in the base harness venv (numpy 1.26 / robosuite 1.4). So the policy runs in the
isolated `/opt/libero-venv` behind `scripts/molmoact2_policy_server.py`, and this adapter is
a thin localhost-HTTP client: it forwards the raw robosuite obs the server needs
(agentview + wrist image, eef pos/quat, gripper qpos) and receives the OSC-delta chunk.

Selected at runtime via POLICY_FACTORY=molmoact2_libero_policy:build_policy.

Env: MM2_SERVER_PORT (8790), MM2_SERVER_TIMEOUT (server-ready wait, s; default 900),
REPLAN_STEPS (chunk replay length; large -> replay the whole model chunk like lerobot-eval),
NUM_STEPS_WAIT (episode-start settle no-ops).
"""
import base64
import json
import os
import time
import urllib.request

import numpy as np

from sim_libero.policy import Policy

PORT = int(os.environ.get("MM2_SERVER_PORT") or "8790")
BASE = f"http://127.0.0.1:{PORT}"


def _post(path, obj, timeout=120):
    data = json.dumps(obj).encode()
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _health():
    try:
        with urllib.request.urlopen(BASE + "/health", timeout=5) as r:
            return json.loads(r.read().decode())
    except Exception:  # noqa: BLE001 - server not up yet
        return None


class MolmoAct2LiberoPolicy(Policy):
    name = "molmoact2"

    def __init__(self):
        # MolmoAct2 predicts a full action chunk and lerobot-eval executes the whole chunk
        # before re-planning, so replay the entire returned chunk (large replan_steps) unless
        # overridden. num_steps_wait settles the scene at episode start.
        self.replan_steps = int(os.environ.get("REPLAN_STEPS") or 256)
        self.num_steps_wait = int(os.environ.get("NUM_STEPS_WAIT") or 10)

    def reset(self, instruction):
        _post("/reset", {"instruction": instruction})

    @staticmethod
    def _img_b64(arr):
        a = np.ascontiguousarray(arr, dtype=np.uint8)  # raw HWC (lerobot feeds raw pixels)
        return base64.b64encode(a.tobytes()).decode(), list(a.shape)

    def predict_action_chunk(self, obs, instruction):
        img_b64, shape = self._img_b64(obs["agentview_image"])
        wrist_b64, _ = self._img_b64(obs["robot0_eye_in_hand_image"])
        payload = {
            "instruction": instruction,
            "image": img_b64, "image2": wrist_b64, "img_shape": shape,
            "eef_pos": np.asarray(obs["robot0_eef_pos"], dtype=np.float64).reshape(-1)[:3].tolist(),
            "eef_quat": np.asarray(obs["robot0_eef_quat"], dtype=np.float64).reshape(-1)[:4].tolist(),
            "gripper_qpos": np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64).reshape(-1)[:2].tolist(),
        }
        out = _post("/act", payload)
        return np.asarray(out["action"], dtype=np.float32).reshape(-1, 7)


def build_policy():
    timeout = float(os.environ.get("MM2_SERVER_TIMEOUT") or "900")
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        h = _health()
        if h is not None:
            if h.get("ready"):
                print(f"[molmoact2_libero_policy] bridge ready at {BASE}", flush=True)
                return MolmoAct2LiberoPolicy()
            if h.get("error"):
                raise RuntimeError(f"MolmoAct2 policy server failed to load:\n{h['error']}")
            last = "loading"
        time.sleep(3)
    raise RuntimeError(
        f"MolmoAct2 policy server not ready at {BASE} within {timeout:.0f}s (state={last}). "
        f"Start it in /opt/libero-venv (the demos do this): "
        f"/opt/libero-venv/bin/python /ryzers/molmoact2_policy_server.py")
