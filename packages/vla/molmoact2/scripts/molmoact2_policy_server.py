# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""MolmoAct2 LIBERO policy server (bridge side of the shared simulation/libero port).

MolmoAct2's LIBERO policy is the allenai *lerobot* stack (lerobot 0.5.1 + transformers 5.3),
which hard-pins numpy>=2; the shared `simulation/libero` base is numpy 1.26 / robosuite 1.4.
The two cannot share one interpreter, so the policy runs here in the isolated
`/opt/libero-venv` and the shared harness (base venv) drives it over localhost HTTP via
`adapters/molmoact2_libero_policy.py`.

This server reuses the *validated* lerobot eval pipeline verbatim -- `make_policy` /
`make_pre_post_processors` / `make_env_pre_post_processors` and the same
`preprocess_observation -> env_pre -> preprocessor -> select_action -> postprocessor`
transform `lerobot.scripts.lerobot_eval.rollout` applies -- so closed-loop behaviour
matches `lerobot-eval --policy.type=molmoact2 --env.type=libero`. Only the env + episode
loop live on the harness side.

Protocol (JSON over HTTP):
  GET  /health          -> {"ready": bool, "error": str|null}
  POST /reset  {instruction}                         -> clears action queue + depth cache
  POST /act    {instruction, image, image2, img_shape, eef_pos, eef_quat, gripper_qpos}
                                                      -> {"action": [[7]*T]}  (OSC delta chunk)

Env: CKPT (allenai/MolmoAct2-Think-LIBERO), THINK (1), SUITE (libero_object),
NUM_STEPS (flow-matching steps; blank=default), MM2_SERVER_PORT (8790).
"""
import base64
import json
import os
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

CKPT = os.environ.get("CKPT") or "allenai/MolmoAct2-Think-LIBERO"
THINK = (os.environ.get("THINK", "1") == "1")
SUITE = os.environ.get("SUITE") or "libero_object"
PORT = int(os.environ.get("MM2_SERVER_PORT") or "8790")

_STATE = {"policy": None, "pre": None, "post": None, "env_pre": None,
          "ready": False, "error": None}
_LOCK = threading.Lock()


def _build_policy():
    import draccus
    from lerobot.configs.eval import EvalPipelineConfig
    from lerobot.envs.factory import make_env_pre_post_processors
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    depth = "True" if THINK else "False"
    args = [
        "--policy.type=molmoact2", f"--policy.checkpoint_path={CKPT}",
        "--policy.inference_action_mode=continuous",
        f"--policy.enable_depth_reasoning={depth}", f"--policy.enable_adaptive_depth={depth}",
        "--policy.enable_cuda_graph=False", "--policy.norm_tag=libero",
        "--policy.device=cuda", "--env.type=libero",
        f"--env.task={SUITE}", "--env.task_ids=[0]",
        "--eval.batch_size=1", "--eval.n_episodes=1", "--output_dir=/tmp/mm2_server",
    ]
    if os.environ.get("NUM_STEPS"):
        args.append(f"--policy.num_steps={os.environ['NUM_STEPS']}")

    cfg = draccus.parse(EvalPipelineConfig, args=args)
    policy = make_policy(cfg=cfg.policy, env_cfg=cfg.env, rename_map=cfg.rename_map)
    policy.eval()
    pre, post = make_pre_post_processors(
        policy_cfg=cfg.policy, pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides={
            "device_processor": {"device": str(policy.config.device)},
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        },
    )
    env_pre, _env_post = make_env_pre_post_processors(env_cfg=cfg.env, policy_cfg=cfg.policy)
    return policy, pre, post, env_pre


def _load_async():
    try:
        policy, pre, post, env_pre = _build_policy()
        with _LOCK:
            _STATE.update(policy=policy, pre=pre, post=post, env_pre=env_pre, ready=True)
        print(f"[mm2_server] policy ready (ckpt={CKPT}, think={THINK}, suite={SUITE})", flush=True)
    except Exception as e:  # noqa: BLE001
        _STATE["error"] = f"{e}\n{traceback.format_exc()}"
        print("[mm2_server] policy load FAILED:\n" + _STATE["error"], flush=True)


def _observation_from_payload(p):
    """Rebuild lerobot's LiberoEnv._format_raw_obs output (pixels_agent_pos, single env)
    from the transmitted raw robosuite fields, then run lerobot's preprocess_observation.
    Only eef pos/quat (xyzw) + gripper qpos are needed downstream (LiberoProcessorStep
    -> observation.state = [eef_pos(3), quat2axisangle(3), gripper_qpos(2)])."""
    from lerobot.envs.utils import preprocess_observation

    def _img(b64, shape):
        return np.frombuffer(base64.b64decode(b64), dtype=np.uint8).reshape(shape)

    # lerobot-eval feeds a vector env (n_envs=1), so robot_state arrays arrive
    # batched as (1, N); preprocess_observation only batches pixels, not robot_state,
    # and LiberoProcessorStep._quat2axisangle requires (B, 4). Add the leading batch
    # axis here so a single observation matches the n_envs=1 eval exactly.
    obs = {
        "pixels": {
            "image": _img(p["image"], p["img_shape"]),
            "image2": _img(p["image2"], p["img_shape"]),
        },
        "robot_state": {
            "eef": {
                "pos": np.asarray(p["eef_pos"], dtype=np.float64).reshape(1, 3),
                "quat": np.asarray(p["eef_quat"], dtype=np.float64).reshape(1, 4),
                "mat": None,
            },
            "gripper": {"qpos": np.asarray(p["gripper_qpos"], dtype=np.float64).reshape(1, 2), "qvel": None},
            "joints": {"pos": None, "vel": None},
        },
    }
    return preprocess_observation(obs)


def _predict_chunk(payload):
    """Return the full OSC-delta chunk [T, 7] for one observation, produced exactly as
    lerobot-eval would (drain the policy's per-episode action queue for this prediction)."""
    import torch

    policy, pre, post, env_pre = (_STATE["policy"], _STATE["pre"], _STATE["post"], _STATE["env_pre"])
    obs = _observation_from_payload(payload)
    obs["task"] = [payload.get("instruction", "")]
    obs = env_pre(obs)   # robot_state -> observation.state (+ image passthrough)
    obs = pre(obs)       # normalize + move to device -> policy batch

    with _LOCK:
        with torch.no_grad():
            policy._action_queues.clear()          # force a fresh chunk prediction
            actions = [post(policy.select_action(obs))]   # predicts, enqueues, pops first
            while policy._action_queues.get(0):           # drain the rest (no re-predict)
                actions.append(post(policy.select_action(obs)))
    chunk = [np.asarray(a.detach().to("cpu").float().numpy()).reshape(-1)[:7] for a in actions]
    return np.stack(chunk).astype(np.float32)


def _reset():
    policy = _STATE["policy"]
    with _LOCK:
        try:
            policy.reset()
        except Exception:  # noqa: BLE001
            pass
        # policy.reset() only clears the action queue; drop the per-episode depth/spatial
        # plan cache too so a new instruction re-plans (mirrors the interactive server).
        for attr in ("_depth_caches", "_last_depth_video_codes_by_batch"):
            cache = getattr(policy, attr, None)
            if isinstance(cache, dict):
                cache.clear()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"ready": bool(_STATE["ready"]), "error": _STATE["error"]})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(n).decode()) if n else {}
        except Exception as e:  # noqa: BLE001
            self._json(400, {"error": f"bad json: {e}"})
            return
        if not _STATE["ready"]:
            self._json(503, {"error": _STATE["error"] or "policy still loading"})
            return
        try:
            if self.path == "/reset":
                _reset()
                self._json(200, {"ok": True})
            elif self.path == "/act":
                chunk = _predict_chunk(payload)
                self._json(200, {"action": chunk.tolist()})
            else:
                self._json(404, {"error": "not found"})
        except Exception as e:  # noqa: BLE001
            self._json(500, {"error": f"{e}", "trace": traceback.format_exc()})


def main():
    threading.Thread(target=_load_async, daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"[mm2_server] listening on 127.0.0.1:{PORT} (loading policy in background)", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
