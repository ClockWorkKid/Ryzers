# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Headless PushT closed-loop rollout against a running VERA policy server (gfx1151 / ROCm).

Headless port of `examples/pusht_dfot_stack.ipynb` (cells 1-6): connects to the DFoT+Jacobian
policy served by `vera.server.start_vera_server --embodiment pusht`, then drives the vendored
PushT env with `PushTRunner` + a `RemotePolicy` (all inference runs server-side on the GPU;
this process only steps the sim). Reset states come from the PushT replay buffer zarr. Writes
a JSON summary; the runner writes per-episode mp4s under OUT_DIR (rule 2.b: generative sim
rollout, no GT reference to overlay).

Env knobs (all optional except ZARR_PATH):
  VERA_HOST/VERA_PORT/VIS_PORT   server endpoint (default 127.0.0.1:8820, vis 8821)
  ZARR_PATH                      pusht_cchi_v7_replay.zarr (reset states)   [required]
  OUT_DIR                        output dir (default /outputs/vera_pusht)
  FRAME_INDICES                  comma list of start-state indices, or "none" to sweep N_STATES
  N_REPEATS / N_STATES           repeats per state / states to sweep when FRAME_INDICES=none
  HORIZON / RENDER_SIZE / SEED / SUCCESS_THRESHOLD
"""
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch


def _env(name: str, default: str) -> str:
    # The ryzers run script threads knobs through as `-e VAR=${VAR:-}`, so an unset host var
    # arrives as a present-but-EMPTY string (which os.environ.get would return over the default).
    # Treat empty/whitespace as "use default" so a bare `ryzers run /ryzers/demos/demo_pusht.sh`
    # works without having to export every knob.
    v = os.environ.get(name, default)
    return default if v is None or v.strip() == "" else v


HOST = _env("VERA_HOST", "127.0.0.1")
PORT = int(_env("VERA_PORT", "8820"))
VIS_PORT = int(_env("VIS_PORT", "8821"))
ZARR_PATH = _env("ZARR_PATH", "/models/pusht/pusht_cchi_v7_replay.zarr")
OUT_DIR = _env("OUT_DIR", "/outputs/vera_pusht")
RENDER_SIZE = int(_env("RENDER_SIZE", "252"))
HORIZON = int(_env("HORIZON", "200"))
SUCCESS_THRESHOLD = float(_env("SUCCESS_THRESHOLD", "0.9"))
SEED = int(_env("SEED", "42"))
N_REPEATS = int(_env("N_REPEATS", "3"))
N_STATES = int(_env("N_STATES", "100"))
ENV_THRESHOLD = 0.82  # env normalizes reward = clip(coverage / 0.82, 0, 1)

_fi = os.environ.get("FRAME_INDICES", "3664").strip().lower()
FRAME_INDICES = None if _fi in ("none", "", "all") else [int(x) for x in _fi.split(",") if x != ""]


def _seed_everything(s: int) -> None:
    torch.manual_seed(s)
    np.random.seed(s)
    random.seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def main() -> int:
    import zarr
    from vera.server.protocol.websocket_policy_client import WebsocketClientPolicy
    from vera.env_runner.pusht_runner import PushtRunnerCfg, PushTRunner
    from vera.controller.run_mimicgen_eval import RemotePolicy  # planner-agnostic remote BasePolicy

    if not Path(ZARR_PATH).exists():
        print(f"FAIL: missing PushT replay zarr: {ZARR_PATH}", file=sys.stderr)
        return 2

    client = WebsocketClientPolicy(host=HOST, port=PORT)
    meta = client.get_server_metadata()
    view_keys = list(meta["view_keys"])
    context_frames = int(meta.get("context_frames", 9))
    print(f"planner (DFoT): {meta.get('planner_model')} | IDM: {meta.get('idm_model')}")
    print(f"views: {view_keys} | context_frames: {context_frames}")
    print(f"action_space: {meta.get('action_space')} | action_dim: {meta.get('action_dim')}")
    print(f"live viewer: http://localhost:{VIS_PORT}/")

    runner_cfg = PushtRunnerCfg(
        env_name="pusht", n_repeat=1, num_env_train=1, num_env_eval=0,
        max_episode_steps=HORIZON, action_scale=1.0,
        output_dir=OUT_DIR, save_videos=True, save_trajectory=False, save_rrd=False, video_fps=5,
    )
    runner = PushTRunner(runner_cfg, device=torch.device("cpu"))
    if getattr(runner, "env", None) is None:
        runner.setup_env()

    remote = RemotePolicy(client, view_keys=view_keys,
                          view_widths=[RENDER_SIZE] * len(view_keys),
                          context_frames=context_frames, prompt=None)
    print("runner + remote policy ready")

    group = zarr.open(ZARR_PATH, mode="r")
    states = group["data"]["state"]
    n_total = states.shape[0]
    if FRAME_INDICES is not None:
        indices = [int(i) for i in FRAME_INDICES]
    else:
        indices = np.linspace(0, n_total - 1, N_STATES, dtype=int).tolist()
    print(f"rolling out {len(indices)} state(s) x {N_REPEATS} repeat(s)")

    rows = []
    for fi in indices:
        for rep in range(1, N_REPEATS + 1):
            _seed_everything(SEED)
            reset_to_state = np.asarray(states[fi], dtype=np.float32)
            out = runner.run(remote, options={"reset_to_state": reset_to_state},
                             run_tag=f"state_{fi}_rep{rep}")
            mrm = float(out.get("max_reward_mean", 0.0))
            coverage = min(mrm, 1.0) * ENV_THRESHOLD
            ret = float(out["train_returns"][0]) if len(out.get("train_returns", [])) else 0.0
            success = int(mrm >= SUCCESS_THRESHOLD)
            rows.append({"frame_idx": fi, "repeat": rep, "max_reward": mrm, "coverage": coverage,
                         "return": ret, "success": success, "save_dir": out.get("save_dir")})
            print(f"  state {fi:>6d} rep{rep}: max_reward={mrm:.4f} coverage={coverage:.4f} "
                  f"return={ret:.4f} success={success}", flush=True)

    sr = 100.0 * np.mean([r["success"] for r in rows]) if rows else 0.0
    tp = 100.0 * np.mean([r["max_reward"] for r in rows]) if rows else 0.0
    cov = 100.0 * np.mean([r["coverage"] for r in rows]) if rows else 0.0
    print("=" * 60)
    print(f"PushT SR = {sr:.1f}%  (max_reward >= {SUCCESS_THRESHOLD}, n={len(rows)})")
    print(f"mean max normalized reward = {tp:.1f}% | mean implied coverage = {cov:.1f}%")
    print("=" * 60)

    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    summary = {"host": HOST, "port": PORT, "zarr": ZARR_PATH, "horizon": HORIZON,
               "render_size": RENDER_SIZE, "seed": SEED, "success_threshold": SUCCESS_THRESHOLD,
               "n": len(rows), "sr_percent": sr, "mean_max_reward_percent": tp,
               "mean_coverage_percent": cov, "rows": rows}
    with open(Path(OUT_DIR) / "pusht_closedloop_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {Path(OUT_DIR) / 'pusht_closedloop_summary.json'}")
    print("PUSHT_CLOSEDLOOP_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
