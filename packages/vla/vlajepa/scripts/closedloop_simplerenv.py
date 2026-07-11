# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Closed-loop SimplerEnv (WidowX / BridgeData v2) rollouts driven by VLA-JEPA on ROCm.

Steps the ManiSkill3 real2sim digital twin (CPU PhysX + SAPIEN offscreen Vulkan on gfx1151):
per trial, reset the env, then repeatedly predict an action and execute its first
``replan_steps`` rows until task success or ``max_steps``. The VLA-JEPA SimplerEnv adapter
re-predicts + temporally ensembles every step (replan_steps=1), matching the upstream
``examples/SimplerEnv/eval_files`` cadence so success rates are comparable to the paper.

The policy is loaded through the model-agnostic ``sim_simplerenv`` seam (POLICY_FACTORY,
default ``vlajepa_simplerenv_policy:build_policy``).

Per workspace rule 2.b (fully generative rollout, no per-frame GT): saved videos are the
single-column third-person rollout, not a GT-vs-pred two-column layout.

Env (see config.yaml):
  TASKS       : comma list of SimplerEnv task names (default: the 4 WidowX/Bridge tasks)
  NUM_TRIALS  : rollouts per task, each from a distinct seed (default 5)
  SEED        : base env seed (default 0); trial ep uses SEED+ep
  MAX_STEPS   : cap env steps per episode (default: env spec's max_episode_steps)
  SAVE_VIDEO  : 1/true (default) writes rollout MP4s; NUM_VIDEOS caps mp4s/task (default 3)
  TAG         : output subdir under /outputs/closedloop (default cl_simplerenv)
  POLICY_FACTORY, and the vlajepa_simplerenv_policy knobs (CKPT_REL, DTYPE, IMAGE_SIZE, ...).
"""
import json
import os
import sys

import imageio
import numpy as np

from sim_simplerenv import simplerenv_env as se
from sim_simplerenv.policy import load_policy

TASKS = [t.strip() for t in (os.environ.get("TASKS") or ",".join(se.WIDOWX_TASKS)).split(",") if t.strip()]
SEED = int(os.environ.get("SEED") or "0")
NUM_TRIALS = int(os.environ.get("NUM_TRIALS") or "5")
SAVE_VIDEO = (os.environ.get("SAVE_VIDEO", "true").lower() not in ("0", "false", "no"))
NUM_VIDEOS = int(os.environ.get("NUM_VIDEOS") or "3")
TAG = os.environ.get("TAG") or "cl_simplerenv"
OUT = os.path.join(os.environ.get("OUT_DIR", "/outputs"), "closedloop", TAG)
MAX_STEPS = int(os.environ["MAX_STEPS"]) if os.environ.get("MAX_STEPS") else None


def run_episode(env, policy, seed, max_steps, save_frames):
    """One closed-loop rollout from a fixed seed; returns (success, frames, instruction)."""
    obs, _ = env.reset(seed=seed)
    instruction = se.get_instruction(env)
    policy.reset(instruction)
    replan = int(getattr(policy, "replan_steps", 1))
    wait = int(getattr(policy, "num_steps_wait", 0))
    frames, ok, t = [], False, 0
    while t < max_steps + wait:
        if t < wait:
            obs, _r, term, trunc, info = env.step(se.get_dummy_action())
            t += 1
            if term or trunc:
                break
            continue
        if save_frames:
            frames.append(se.get_image(env, obs))
        chunk = np.asarray(policy.predict_action_chunk(obs, instruction))
        term = trunc = False
        for a in chunk[:replan]:
            obs, _r, term, trunc, info = env.step(np.asarray(a, dtype=np.float32))
            t += 1
            if se.is_success(info, term):
                ok = True
            if term or trunc:
                break
        if ok or term or trunc:
            break
    return bool(ok), frames, instruction


def main() -> int:
    import torch

    print(f"torch          : {torch.__version__} hip={torch.version.hip}")
    if not (torch.version.hip and torch.cuda.is_available()):
        print("FAIL: need a ROCm device.", file=sys.stderr)
        return 1
    print(f"device[0]      : {torch.cuda.get_device_name(0)}")
    os.makedirs(OUT, exist_ok=True)
    print(f"tasks          : {TASKS}  trials/task={NUM_TRIALS}  seed={SEED}", flush=True)

    policy = load_policy()  # builds VLA-JEPA once via POLICY_FACTORY

    rows, tot_s, tot_n = [], 0, 0
    for task in TASKS:
        env, _obs, instr = se.build_env(task, seed=SEED)
        max_steps = MAX_STEPS if MAX_STEPS is not None else se.get_max_steps(env)
        print(f"########## {task}: {instr!r}  (max_steps={max_steps}) ##########", flush=True)
        s = n = 0
        for ep in range(NUM_TRIALS):
            save = SAVE_VIDEO and ep < NUM_VIDEOS
            ok, frames, instruction = run_episode(env, policy, SEED + ep, max_steps, save)
            s += int(ok)
            n += 1
            if save and frames:
                suffix = "success" if ok else "failure"
                imageio.mimwrite(
                    os.path.join(OUT, f"{task}_ep{ep}_{suffix}.mp4"),
                    [np.asarray(f) for f in frames], fps=10)
            print(f"  {task} ep {ep}: {'SUCCESS' if ok else 'fail'}  (running {s}/{n})", flush=True)
        env.close()
        rate = round(100.0 * s / n, 2) if n else 0.0
        json.dump({"task": task, "successes": s, "total_episodes": n,
                   "success_rate_pct": rate, "instruction": instr},
                  open(os.path.join(OUT, f"{task}_results.json"), "w"), indent=2)
        rows.append((task, s, n, instr))
        tot_s += s
        tot_n += n
        print(f"  => {task}: {s}/{n} ({rate:.1f}%)", flush=True)

    overall = round(100.0 * tot_s / tot_n, 2) if tot_n else 0.0
    summary = {
        "overall_successes": tot_s, "overall_episodes": tot_n,
        "overall_success_rate_pct": overall,
        "per_task": [{"task": t, "successes": ss, "episodes": nn,
                      "success_rate_pct": round(100.0 * ss / nn, 2) if nn else 0.0, "instruction": d}
                     for t, ss, nn, d in rows],
    }
    json.dump(summary, open(os.path.join(OUT, "success_summary.json"), "w"), indent=2)
    print(f"OVERALL        : {tot_s}/{tot_n} ({overall:.1f}%)")
    print(f"summary        : {os.path.join(OUT, 'success_summary.json')}")
    print("PASS: VLA-JEPA closed-loop SimplerEnv OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
