# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Model-agnostic websocket policy: drive the sim from ANY model served over the protocol.

This is a generalized port of VERA's upstream `run_mimicgen_eval.RemotePolicy` (rule 2.1). A
model runs as a policy server (obs->action-chunk over the websocket protocol in
`vera.server.protocol`); this `Policy` connects to it, reads the server metadata (view_keys,
context_frames), and bridges the runner's per-step `predict_action` to the server's chunked
`infer`: it accumulates a rolling context window of the runner's (already view-concatenated)
rgb, calls `infer` only when the local action queue is empty, and pops one action per step.

Because the seam is the wire protocol, the harness is model-agnostic: VERA's WAN+IDM server,
or any other model implementing the same `get_server_metadata()/infer()` endpoints, plugs in
by pointing this policy at `--host/--port` (POLICY_HOST / POLICY_PORT env, or a model layer's
own `build_policy` factory). No model code lives in the simulator package.
"""
from __future__ import annotations

import os
import time
import uuid
from collections import deque
from typing import List, Optional

import numpy as np

from sim_mimicgen.policy import Policy, PolicyObservation, PolicyOutput


class RemoteWebsocketPolicy(Policy):
    """A seam Policy whose inference runs on a remote websocket model server."""

    cfg = None
    device = None
    name = "remote"

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8800,
        *,
        context_frames: Optional[int] = None,
        render_size: Optional[int] = None,
        prompt: Optional[str] = None,
        verbose: bool = True,
    ):
        # Imported here so the module loads even where the ws client isn't installed.
        from vera.server.protocol.websocket_policy_client import WebsocketClientPolicy

        self._client = WebsocketClientPolicy(host=host, port=port)
        meta = self._client.get_server_metadata()
        self.view_keys = list(meta["view_keys"])
        # Use the model's real context length (server-advertised 1+(N-1)*stride) unless overridden.
        self.context_frames = int(context_frames) if context_frames else int(meta.get("context_frames", 9))
        self.render_size = int(render_size) if render_size else 128
        self._view_widths = [self.render_size] * len(self.view_keys)
        self._window: deque = deque(maxlen=int(self.context_frames))
        self._queue: deque = deque()
        self._session = str(uuid.uuid4())
        self.prompt = prompt
        self._verbose = bool(verbose)
        self._chunk = 0
        self._step = 0
        self._meta = meta

    @staticmethod
    def _to_uint8(rgb) -> np.ndarray:
        a = np.asarray(rgb)
        if a.ndim == 4:
            a = a[0]
        if np.issubdtype(a.dtype, np.floating):
            a = (np.clip(a, 0.0, 1.0) * 255).astype(np.uint8)
        return np.ascontiguousarray(a.astype(np.uint8))

    def reset(self) -> None:
        self._window.clear()
        self._queue.clear()
        self._session = str(uuid.uuid4())
        self._chunk = 0
        self._step = 0
        self._client.reset({"session_id": self._session, "reason": "eval_episode"})

    def warmup_obs(self, obs: PolicyObservation) -> None:
        self._window.append(self._to_uint8(obs.rgb))  # fill context, no inference

    def predict_action(self, obs: PolicyObservation) -> PolicyOutput:
        self._step += 1
        self._window.append(self._to_uint8(obs.rgb))
        if not self._queue:
            context_rgb = np.stack(list(self._window), axis=0)  # (T,H,W,3) uint8
            req = {
                "context_rgb": context_rgb,
                "view_keys": list(obs.view_keys or self.view_keys),
                "view_widths": list(obs.view_widths or self._view_widths),
                "session_id": self._session,
            }
            if self.prompt is not None:
                req["prompt"] = self.prompt
            if self._verbose:
                print(f"  chunk {self._chunk + 1:>3} (env step {self._step:>3}): "
                      f"dreaming + denoising on server…", end="", flush=True)
            _t0 = time.time()
            out = self._client.infer(req)
            actions = np.asarray(out["action"], dtype=np.float32)
            for row in actions:
                self._queue.append(row)
            self._chunk += 1
            if self._verbose:
                print(f"\r  chunk {self._chunk:>3} (env step {self._step:>3}): dream done in "
                      f"{time.time() - _t0:4.1f}s → committing {len(actions)} actions "
                      f"(|a|={np.abs(actions).mean():.3f}) ", flush=True)
        action = self._queue.popleft()
        return PolicyOutput(action=action[None, :], info=None)  # (1, D) for the env step

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass


def build_policy() -> Policy:
    """POLICY_FACTORY entry: RemoteWebsocketPolicy from POLICY_HOST/POLICY_PORT env."""
    return RemoteWebsocketPolicy(
        host=os.environ.get("POLICY_HOST", "127.0.0.1"),
        port=int(os.environ.get("POLICY_PORT", os.environ.get("PORT", 8800))),
        context_frames=(int(os.environ["CONTEXT_FRAMES"]) if os.environ.get("CONTEXT_FRAMES") else None),
        render_size=(int(os.environ["RENDER_SIZE"]) if os.environ.get("RENDER_SIZE") else None),
        prompt=os.environ.get("PROMPT") or None,
    )
