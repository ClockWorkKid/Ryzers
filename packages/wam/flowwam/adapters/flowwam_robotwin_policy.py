# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""FlowWAM RoboTwin policy-factory adapter for the interactive sim harness.

Wraps the SAME dual-stream flow-action model + checkpoint the closed-loop demo uses behind
the model-agnostic ``sim_robotwin.Policy`` seam, selected via
``POLICY_FACTORY=flowwam_robotwin_policy:build_policy``.

FlowWAM's control path is a websocket ``flow_action_server`` (dual-stream Wan DiT generates
flow-conditioned video latents; an IDM action expert decodes a 14-DoF qpos chunk) driven by
the upstream ``robotwin_policy`` client. ``build_policy`` launches that server exactly like
``demos/demo_closedloop_robotwin.sh`` and connects the lightweight client
(``flowwam_model.FlowWAMPolicy``), so the interactive showcase and the closed loop drive an
identical model. ``predict_action_chunk`` mirrors the upstream ``deploy_policy`` seam: encode
the RoboTwin obs, push it to the server, and return the future qpos rows (the GT-anchored
``action[0]`` is skipped).

Env: CKPT, ACTION_NORM_PATH, LOCAL_MODEL_PATH (/models/flowwam), FLOWWAM_FULL_REPO
(/repos/flowwam-full), FLOW_SERVER_HOST (0.0.0.0), FLOW_SERVER_PORT (8000), EXECUTE_WINDOW
(25), VIDEO_INFERENCE_STEPS (25), ACTION_INFERENCE_STEPS (50), FLOWWAM_PYTHON, OUT_DIR.
"""
import atexit
import os
import re
import signal
import subprocess
import time

import numpy as np

from sim_robotwin.policy import Policy

DEFAULT_CKPT = "/models/flowwam/robotwin/flowwam_robotwin.safetensors"
DEFAULT_NORM = "/models/flowwam/robotwin/flowwam_robotwin_action_norm_stats.npz"
_READY_RE = re.compile(r"serving on ws|Server ready|Listening", re.IGNORECASE)


def _env(name, default=""):
    val = os.environ.get(name)
    return val if val not in (None, "") else default


def _encode_obs(observation):
    """RoboTwin obs -> ([head, right, left] uint8 RGB, (14,) qpos). Mirrors the upstream
    robotwin_policy/deploy_policy.encode_obs so interactive == closed-loop conditioning."""
    rgb = [
        observation["observation"]["head_camera"]["rgb"],
        observation["observation"]["right_camera"]["rgb"],
        observation["observation"]["left_camera"]["rgb"],
    ]
    state = observation["joint_action"]["vector"]
    return rgb, state


def _start_server(full_repo, ckpt, norm, local_model_path, host, port, log_path):
    """Launch the upstream flow_action_server via start_server.sh (same route as closed-loop:
    PYTHON=opt_python.sh arms the gfx1151 speedups, then runs the server verbatim)."""
    start_sh = os.path.join(full_repo, "inference", "start_server.sh")
    if not os.path.isfile(start_sh):
        raise FileNotFoundError(
            f"{start_sh} not found -- clone the upstream FlowWAM repo (see demo script)")
    env = dict(os.environ)
    env.update(
        CHECKPOINT=ckpt,
        ACTION_NORM_PATH=norm,
        LOCAL_MODEL_PATH=local_model_path,
        HOST=host,
        PORT=str(port),
        DEVICE="cuda",
        VIDEO_INFERENCE_STEPS=_env("VIDEO_INFERENCE_STEPS", "25"),
        ACTION_INFERENCE_STEPS=_env("ACTION_INFERENCE_STEPS", "50"),
        PYTHON=_env("FLOWWAM_PYTHON", "/ryzers/scripts/opt_python.sh"),
        PYTHONPATH=full_repo + ":" + env.get("PYTHONPATH", ""),
    )
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    log = open(log_path, "wb")
    print(f"[flowwam_robotwin_policy] launching flow_action_server on ws://{host}:{port} "
          f"(log: {log_path})", flush=True)
    proc = subprocess.Popen(
        ["bash", start_sh], env=env, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    def _cleanup():
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            pass
    atexit.register(_cleanup)
    # SIGTERM handler is best-effort: signal.signal() only works from the main thread, and
    # build_policy runs in the interactive server's engine thread, so guard it.
    try:
        signal.signal(signal.SIGTERM, lambda *_: (_cleanup(), os._exit(0)))
    except (ValueError, RuntimeError):
        pass
    return proc, log_path


def _wait_ready(proc, log_path, timeout):
    """Block until the server logs a serving line, or it dies / times out."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            tail = ""
            try:
                with open(log_path, "r", errors="replace") as f:
                    tail = "".join(f.readlines()[-30:])
            except Exception:
                pass
            raise RuntimeError(f"flow_action_server exited early (rc={proc.returncode}).\n{tail}")
        try:
            with open(log_path, "r", errors="replace") as f:
                if _READY_RE.search(f.read()):
                    return
        except FileNotFoundError:
            pass
        time.sleep(3)
    raise TimeoutError(f"flow_action_server not ready within {timeout}s (log: {log_path})")


class FlowwamRoboTwinPolicy(Policy):
    name = "flowwam"
    action_type = "qpos"

    def __init__(self, model, replan_steps):
        self.model = model  # flowwam_model.FlowWAMPolicy (ws client to flow_action_server)
        self.replan_steps = int(replan_steps)

    def reset(self, instruction):
        self.model.reset()
        self.model.set_language(instruction)

    def predict_action_chunk(self, obs, instruction):
        if self.model.instruction is None:
            self.model.set_language(instruction)
        rgb, state = _encode_obs(obs)
        self.model.update_observation_window(rgb, state)
        actions = self.model.get_action()  # (1 + EXECUTE_WINDOW, 14) absolute qpos
        # action[0] is the GT anchor (current qpos); execute only the future rows.
        return np.asarray(actions[self.model.pixel_prefix:], dtype=np.float32)


def build_policy():
    full_repo = _env("FLOWWAM_FULL_REPO", "/repos/flowwam-full")
    ckpt = _env("CKPT", DEFAULT_CKPT)
    norm = _env("ACTION_NORM_PATH", DEFAULT_NORM)
    local_model_path = _env("LOCAL_MODEL_PATH", _env("FLOWWAM_MODEL_DIR", "/models/flowwam"))
    host = _env("FLOW_SERVER_HOST", "0.0.0.0")
    port = int(_env("FLOW_SERVER_PORT", "8000"))
    execute_window = int(_env("EXECUTE_WINDOW", "25"))
    out_dir = _env("OUT_DIR", "/sim_outputs")
    log_path = os.path.join(out_dir, "interactive", "flow_action_server.log")

    proc, log_path = _start_server(full_repo, ckpt, norm, local_model_path, host, port, log_path)
    _wait_ready(proc, log_path, int(_env("FLOWWAM_SERVER_TIMEOUT", "1800")))

    from flowwam_model import FlowWAMPolicy  # upstream ws client (websockets + msgpack only)
    model = FlowWAMPolicy(
        host=host, port=port,
        action_chunk_size=1 + execute_window,
        task_name=_env("TASK", None) or None,
    )
    print(f"[flowwam_robotwin_policy] policy ready (ckpt={ckpt}, "
          f"execute_window={execute_window})", flush=True)
    return FlowwamRoboTwinPolicy(model, replan_steps=execute_window)
