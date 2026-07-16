# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Closed-loop RoboLab-AMD evaluation (gate G4).

Drives a real policy (via POLICY_FACTORY, e.g. the Cosmos3-Nano-Policy-DROID websocket
adapter) through one RoboLab task across a sweep of seeds, executing predicted joint-space
chunks with genuine physics + real YCB assets, and records per-episode success + an MP4 of
the composed 3-view. Stops at the first success when STOP_ON_SUCCESS=1 (the pilot gate is a
single successful end-to-end episode); otherwise runs the full seed budget and reports the
success rate. Also logs server-side inference latency when the policy exposes it.

Env: TASK, SEED (first seed), NUM_EPISODES (seed budget), STEPS (per-episode cap),
     STOP_ON_SUCCESS, OUT_DIR, POLICY_FACTORY.
"""
import json
import os
from datetime import datetime

from sim_robolab.envutil import env_int, env_str
from sim_robolab.policy import load_policy
from sim_robolab.render import banner_frame, compose_view, save_mp4
from sim_robolab.rollout import run_episode
from sim_robolab.scene import build_scene


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs)) if xs else None


def main():
    task = env_str("TASK", "BananaInBowl")
    first_seed = env_int("SEED", 0)
    num_episodes = env_int("NUM_EPISODES", 10)
    steps = env_int("STEPS", 0)  # 0 -> task default
    stop_on_success = env_str("STOP_ON_SUCCESS", "1") not in ("0", "", "false", "False")
    out_dir = env_str("OUT_DIR", "/sim_outputs")
    os.makedirs(out_dir, exist_ok=True)

    policy = load_policy()
    pname = getattr(policy, "name", type(policy).__name__)
    print(f"[closedloop] task={task} policy={pname} seeds={first_seed}..{first_seed + num_episodes - 1} "
          f"stop_on_success={stop_on_success}", flush=True)

    results = []
    successes = 0
    ts = datetime.now().strftime("%H%M%S")

    for i in range(num_episodes):
        seed = first_seed + i
        scene = build_scene(task, seed=seed)
        instruction = scene.description
        max_steps = steps if steps > 0 else scene.max_steps

        frames = []

        def on_frame(view, step, replanning, _f=frames, _p=pname, _instr=instruction):
            tag = "THINKING" if replanning else ""
            _f.append(banner_frame(compose_view(view), f"{_p}: {_instr}", 720, tag=tag))

        print(f"[closedloop] seed={seed} rolling (max_steps={max_steps}) ...", flush=True)
        success, model_steps = run_episode(
            scene, policy, instruction, on_frame=on_frame, max_steps=max_steps
        )
        scene.close()

        label = "success" if success else "fail"
        path = os.path.join(out_dir, f"closedloop_{task}_seed{seed}_{label}_{ts}.mp4")
        if frames:
            save_mp4(frames, path, fps=20)
        successes += int(success)
        results.append({
            "seed": seed,
            "success": bool(success),
            "model_calls": model_steps,
            "frames": len(frames),
            "video": path if frames else None,
        })
        print(f"[closedloop] seed={seed} success={success} model_calls={model_steps} "
              f"frames={len(frames)} -> {path}", flush=True)

        if success and stop_on_success:
            print(f"[closedloop] first success at seed={seed}; stopping (single-demo gate).", flush=True)
            break

    attempted = len(results)
    summary = {
        "task": task,
        "policy": pname,
        "attempted": attempted,
        "successes": successes,
        "success_rate": (successes / attempted) if attempted else 0.0,
        "mean_infer_ms": _mean(getattr(policy, "infer_ms_log", []) or []),
        "results": results,
    }
    spath = os.path.join(out_dir, f"closedloop_{task}_summary_{ts}.json")
    with open(spath, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[closedloop] DONE {successes}/{attempted} success; "
          f"mean_infer_ms={summary['mean_infer_ms']}; summary -> {spath}", flush=True)

    try:
        policy_close = getattr(policy, "close", None)
        if callable(policy_close):
            policy_close()
    except Exception:  # noqa: BLE001
        pass


if __name__ == "__main__":
    main()
