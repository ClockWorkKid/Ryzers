# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Closed-loop LIBERO-Plus robustness rollouts driven by VLA-JEPA on ROCm (headless EGL).

LIBERO-Plus expands each LIBERO suite into thousands of *perturbation instances* across 7
robustness dimensions (camera, robot state, language, light, background, noise, layout) and
5 difficulty levels. Evaluation is identical to LIBERO except each task runs a single trial
(its init state encodes one perturbation config). This runner steps the real MuJoCo sim the
same way as ``closedloop_libero.py`` (settle -> chunk-replay) but:

  * selects tasks by perturbation CATEGORY and/or DIFFICULTY via the sim base's
    ``task_classification.json`` loader (``sim_liberoplus.libero_env``),
  * runs NUM_TRIALS=1 per task by default (the LIBERO-Plus convention),
  * caps / randomly samples the selection (MAX_TASKS + SEED) since a full suite is thousands
    of tasks, and
  * aggregates success per dimension and per difficulty level, the way the LIBERO-Plus paper
    reports robustness -- not just an overall number.

The policy is loaded through the model-agnostic ``sim_liberoplus`` seam (POLICY_FACTORY,
default ``vlajepa_liberoplus_policy:build_policy``); preprocessing / action handling is
identical to the LIBERO adapter.

Env (see config.yaml):
  SUITE       : libero_object|libero_goal|libero_spatial|libero_10  (default libero_object)
  CATEGORY    : perturbation dimension (e.g. "Camera Viewpoints"); empty/"all" = every dim
  DIFFICULTY  : difficulty level 1-5; empty/"all" = every level
  MAX_TASKS   : cap number of tasks after filtering (default 40; 0 = all matching)
  SAMPLE      : 1 (default) random-sample MAX_TASKS with SEED; 0 = take the first MAX_TASKS
  NUM_TRIALS  : rollouts per task (default 1, the LIBERO-Plus convention)
  SEED        : env + sampling seed (default 1000)
  MAX_STEPS   : cap env steps per episode (default: per-suite horizon)
  SAVE_VIDEO  : 1/true (default) writes rollout MP4s; NUM_VIDEOS caps total mp4s (default 8)
  TAG         : output subdir under /outputs/liberoplus (default lp_<SUITE>)
  POLICY_FACTORY, and the vlajepa_liberoplus_policy knobs (CKPT_REL, DTYPE, REPLAN_STEPS, ...).
"""
import json
import os
import random
import sys
from collections import defaultdict

import imageio
import numpy as np

from sim_liberoplus.libero_env import (
    LIBERO_ENV_RESOLUTION,
    PERTURBATION_CATEGORIES,
    get_benchmark_dict,
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_max_steps,
    load_task_classification,
    resolve_task_ids,
)
from sim_liberoplus.policy import load_policy

SUITE = os.environ.get("SUITE") or "libero_object"
SEED = int(os.environ.get("SEED") or "1000")
NUM_TRIALS = int(os.environ.get("NUM_TRIALS") or "1")
MAX_TASKS = int(os.environ.get("MAX_TASKS") or "40")  # 0 = all matching
SAMPLE = (os.environ.get("SAMPLE", "1").lower() not in ("0", "false", "no"))
SAVE_VIDEO = (os.environ.get("SAVE_VIDEO", "true").lower() not in ("0", "false", "no"))
NUM_VIDEOS = int(os.environ.get("NUM_VIDEOS") or "8")
TAG = os.environ.get("TAG") or f"lp_{SUITE}"
OUT = os.path.join(os.environ.get("OUT_DIR", "/outputs"), "liberoplus", TAG, SUITE)
MAX_STEPS = int(os.environ["MAX_STEPS"]) if os.environ.get("MAX_STEPS") else None


def _norm_cat(value):
    """Match a CATEGORY env value (space or underscore form, any case) to a canonical name."""
    if not value or value.lower() == "all":
        return None
    key = value.replace("_", " ").strip().lower()
    for c in PERTURBATION_CATEGORIES:
        if c.lower() == key:
            return c
    raise ValueError(f"Unknown CATEGORY {value!r}. Valid: {PERTURBATION_CATEGORIES} (or 'all').")


def _select_entries(classification):
    """Filter + (optionally) sample the suite's perturbation entries by CATEGORY/DIFFICULTY."""
    entries = list(classification.get(SUITE, []))
    cat = _norm_cat(os.environ.get("CATEGORY"))
    if cat is not None:
        entries = [e for e in entries if e.get("category") == cat]
    diff = os.environ.get("DIFFICULTY")
    if diff and diff.lower() != "all":
        d = int(diff)
        entries = [e for e in entries if int(e.get("difficulty_level", -1)) == d]
    if MAX_TASKS and len(entries) > MAX_TASKS:
        if SAMPLE:
            rng = random.Random(SEED)
            entries = rng.sample(entries, MAX_TASKS)
        else:
            entries = entries[:MAX_TASKS]
    return entries, cat, (diff if diff and diff.lower() != "all" else None)


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


def _rate(s, n):
    return round(100.0 * s / n, 2) if n else 0.0


def main() -> int:
    import torch

    print(f"torch          : {torch.__version__} hip={torch.version.hip}")
    if not (torch.version.hip and torch.cuda.is_available()):
        print("FAIL: need a ROCm device.", file=sys.stderr)
        return 1
    print(f"device[0]      : {torch.cuda.get_device_name(0)}")
    os.makedirs(OUT, exist_ok=True)

    classification = load_task_classification()
    entries, cat, diff = _select_entries(classification)
    if not entries:
        print("FAIL: no tasks matched the CATEGORY/DIFFICULTY selection.", file=sys.stderr)
        return 1

    task_suite = get_benchmark_dict()[SUITE]()
    selected = resolve_task_ids(task_suite, entries)  # [(task_id, entry), ...]
    max_steps = MAX_STEPS if MAX_STEPS is not None else get_max_steps(SUITE)
    print(f"suite          : {SUITE}  category={cat or 'ALL'}  difficulty={diff or 'ALL'}")
    print(f"selection      : {len(selected)} tasks  trials/task={NUM_TRIALS}  "
          f"max_steps={max_steps}  (sample={SAMPLE}, seed={SEED})", flush=True)

    policy = load_policy()  # builds VLA-JEPA once via POLICY_FACTORY

    by_cat = defaultdict(lambda: [0, 0])   # category -> [successes, episodes]
    by_diff = defaultdict(lambda: [0, 0])  # difficulty_level -> [successes, episodes]
    per_task, tot_s, tot_n, n_videos = [], 0, 0, 0

    for i, (tid, entry) in enumerate(selected):
        task = task_suite.get_task(tid)
        init_states = task_suite.get_task_init_states(tid)
        env, desc = get_libero_env(task, LIBERO_ENV_RESOLUTION, SEED)
        ecat = entry.get("category", "?")
        ediff = int(entry.get("difficulty_level", -1))
        s = n = 0
        for ep in range(NUM_TRIALS):
            init = init_states[ep % len(init_states)]
            save = SAVE_VIDEO and n_videos < NUM_VIDEOS
            ok, frames = run_episode(env, policy, init, desc, max_steps, save)
            s += int(ok)
            n += 1
            if save and frames:
                suffix = "success" if ok else "failure"
                cat_slug = ecat.replace(" ", "_")
                imageio.mimwrite(
                    os.path.join(OUT, f"{cat_slug}_L{ediff}_task{tid}_ep{ep}_{suffix}.mp4"),
                    [np.asarray(f) for f in frames], fps=20)
                n_videos += 1
        by_cat[ecat][0] += s
        by_cat[ecat][1] += n
        by_diff[ediff][0] += s
        by_diff[ediff][1] += n
        tot_s += s
        tot_n += n
        per_task.append({"task_id": tid, "name": entry.get("name"), "category": ecat,
                         "difficulty_level": ediff, "successes": s, "episodes": n})
        print(f"  [{i + 1}/{len(selected)}] task {tid} [{ecat} L{ediff}]: "
              f"{s}/{n}  (running {tot_s}/{tot_n} = {_rate(tot_s, tot_n):.1f}%)  {desc[:60]}",
              flush=True)
        env.close()

    summary = {
        "suite": SUITE,
        "category_filter": cat or "all",
        "difficulty_filter": diff or "all",
        "num_trials_per_task": NUM_TRIALS,
        "overall_successes": tot_s,
        "overall_episodes": tot_n,
        "overall_success_rate_pct": _rate(tot_s, tot_n),
        "by_category": {k: {"successes": v[0], "episodes": v[1],
                            "success_rate_pct": _rate(v[0], v[1])}
                        for k, v in sorted(by_cat.items())},
        "by_difficulty": {f"L{k}": {"successes": v[0], "episodes": v[1],
                                    "success_rate_pct": _rate(v[0], v[1])}
                          for k, v in sorted(by_diff.items())},
        "per_task": per_task,
    }
    json.dump(summary, open(os.path.join(OUT, "robustness_summary.json"), "w"), indent=2)

    print(f"\nOVERALL        : {tot_s}/{tot_n} ({_rate(tot_s, tot_n):.1f}%)")
    print("by dimension   :")
    for k, v in sorted(by_cat.items()):
        print(f"  {k:<22}: {v[0]}/{v[1]} ({_rate(v[0], v[1]):.1f}%)")
    print("by difficulty  :")
    for k, v in sorted(by_diff.items()):
        print(f"  L{k}: {v[0]}/{v[1]} ({_rate(v[0], v[1]):.1f}%)")
    print(f"summary        : {os.path.join(OUT, 'robustness_summary.json')}")
    print("PASS: VLA-JEPA closed-loop LIBERO-Plus OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
