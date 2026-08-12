# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Capability 3: closed-loop LIBERO rollouts driven by VLA-JEPA on ROCm (headless EGL).

Unlike open-loop replay (which scores predictions against a recorded demo), this steps the
real MuJoCo LIBERO simulator: settle for ``num_steps_wait`` no-ops, then repeatedly predict
an action chunk and execute its first ``replan_steps`` actions until task success or
``max_steps``. Compounding error and grasp dynamics are exercised for real.

The policy is loaded through the model-agnostic ``sim_libero`` seam (POLICY_FACTORY, default
``vlajepa_libero_policy:build_policy``). The episode loop mirrors upstream
``examples/LIBERO/eval_libero.py`` and FastWAM's ``demo_closedloop_libero.sh`` so success
rates are comparable: same 180-deg image preprocessing (via ``sim_libero.get_libero_image``),
same per-trial initial states, same chunk-replay stepping.

Per workspace rule 2.b (fully generative rollout, no per-frame GT): saved videos are the
single-column rollout (agentview) rather than a GT-vs-pred two-column layout.

Env (see config.yaml):
  SUITE       : libero_object|libero_goal|libero_spatial|libero_10  (default libero_object)
  NUM_TASKS   : number of tasks from the suite to evaluate (default 3; ignored if TASK_ID set)
  TASK_ID     : evaluate a single task id only (overrides NUM_TASKS)
  NUM_TRIALS  : rollouts per task, each from a distinct initial state (default 5)
  SEED        : env seed (default 1000)
  MAX_STEPS   : cap env steps per episode (default: per-suite horizon)
  SAVE_VIDEO  : 1/true (default) writes rollout MP4s; NUM_VIDEOS caps mp4s/task (default 3)
  TAG         : output subdir under /outputs/closedloop (default cl_<SUITE>)
  POLICY_FACTORY, and the vlajepa_libero_policy knobs (CKPT_REL, DTYPE, REPLAN_STEPS, ...).
"""
import json
import os
import sys

import imageio
import numpy as np

from sim_libero.libero_env import (
    LIBERO_ENV_RESOLUTION,
    get_benchmark_dict,
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_max_steps,
)
from sim_libero.policy import load_policy

SUITE = os.environ.get("SUITE") or "libero_object"
SEED = int(os.environ.get("SEED") or "1000")
NUM_TRIALS = int(os.environ.get("NUM_TRIALS") or "5")
SAVE_VIDEO = (os.environ.get("SAVE_VIDEO", "true").lower() not in ("0", "false", "no"))
NUM_VIDEOS = int(os.environ.get("NUM_VIDEOS") or "3")
TAG = os.environ.get("TAG") or f"cl_{SUITE}"
OUT = os.path.join(os.environ.get("OUT_DIR", "/outputs"), "closedloop", TAG, SUITE)
MAX_STEPS = int(os.environ["MAX_STEPS"]) if os.environ.get("MAX_STEPS") else None


def run_episode(env, policy, init_state, instruction, max_steps, save_frames):
    """One closed-loop rollout from a fixed initial state; returns (success, frames)."""
    policy.reset(instruction)
    env.reset()
    obs = env.set_init_state(init_state)
    replan = int(getattr(policy, "replan_steps", 5))
    wait = int(getattr(policy, "num_steps_wait", 10))
    pending, frames, done, t = [], [], False, 0
    while t < max_steps + wait:
        if t < wait:  # let dropped objects settle before acting
            obs, _, done, _ = env.step(get_libero_dummy_action())
            t += 1
            continue
        if not pending:
            chunk = np.asarray(policy.predict_action_chunk(obs, instruction))
            pending = [list(a) for a in chunk[:replan]]
        if save_frames:
            frames.append(get_libero_image(obs)["image"])
        obs, _, done, _ = env.step(pending.pop(0))
        t += 1
        if done:
            break
    return bool(done), frames


def main() -> int:
    import torch

    print(f"torch          : {torch.__version__} hip={torch.version.hip}")
    if not (torch.version.hip and torch.cuda.is_available()):
        print("FAIL: need a ROCm device.", file=sys.stderr)
        return 1
    print(f"device[0]      : {torch.cuda.get_device_name(0)}")
    os.makedirs(OUT, exist_ok=True)

    task_suite = get_benchmark_dict()[SUITE]()
    n_tasks = task_suite.n_tasks
    if os.environ.get("TASK_ID"):
        task_ids = [int(os.environ["TASK_ID"])]
    else:
        task_ids = list(range(min(int(os.environ.get("NUM_TASKS") or n_tasks), n_tasks)))
    max_steps = MAX_STEPS if MAX_STEPS is not None else get_max_steps(SUITE)
    print(f"suite          : {SUITE}  tasks={task_ids}  trials/task={NUM_TRIALS}  "
          f"max_steps={max_steps}")

    policy = load_policy()  # builds VLA-JEPA once via POLICY_FACTORY

    rows, tot_s, tot_n = [], 0, 0
    for tid in task_ids:
        task = task_suite.get_task(tid)
        init_states = task_suite.get_task_init_states(tid)
        env, desc = get_libero_env(task, LIBERO_ENV_RESOLUTION, SEED)
        print(f"########## {SUITE} task {tid}: {desc} ##########", flush=True)
        s = n = 0
        for ep in range(NUM_TRIALS):
            init = init_states[ep % len(init_states)]
            save = SAVE_VIDEO and ep < NUM_VIDEOS
            ok, frames = run_episode(env, policy, init, desc, max_steps, save)
            s += int(ok)
            n += 1
            if save and frames:
                suffix = "success" if ok else "failure"
                imageio.mimwrite(
                    os.path.join(OUT, f"task{tid}_ep{ep}_{suffix}.mp4"),
                    [np.asarray(f) for f in frames], fps=20)
            print(f"  task {tid} ep {ep}: {'SUCCESS' if ok else 'fail'}  "
                  f"(running {s}/{n})", flush=True)
        env.close()
        rate = round(100.0 * s / n, 2) if n else 0.0
        json.dump({"task_id": tid, "successes": s, "total_episodes": n,
                   "success_rate_pct": rate, "task_description": desc},
                  open(os.path.join(OUT, f"task{tid}_results.json"), "w"), indent=2)
        rows.append((tid, s, n, desc))
        tot_s += s
        tot_n += n
        print(f"  => task {tid}: {s}/{n} ({rate:.1f}%)", flush=True)

    overall = round(100.0 * tot_s / tot_n, 2) if tot_n else 0.0
    summary = {
        "suite": SUITE, "overall_successes": tot_s, "overall_episodes": tot_n,
        "overall_success_rate_pct": overall,
        "per_task": [{"task_id": t, "successes": ss, "episodes": nn,
                      "success_rate_pct": round(100.0 * ss / nn, 2) if nn else 0.0, "task": d}
                     for t, ss, nn, d in rows],
    }
    json.dump(summary, open(os.path.join(OUT, "success_summary.json"), "w"), indent=2)
    print(f"OVERALL        : {tot_s}/{tot_n} ({overall:.1f}%)")
    print(f"summary        : {os.path.join(OUT, 'success_summary.json')}")
    print("PASS: VLA-JEPA closed-loop LIBERO OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
