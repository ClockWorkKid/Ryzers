# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Environment sign-of-life for the LIBERO-Plus simulator base image (Strix Halo, gfx1151).

Runs inside the built image with NO model. Proves the container has (1) the LIBERO-Plus /
MuJoCo / robosuite stack importing cleanly, (2) the sim_liberoplus harness + built-in
RandomPolicy resolving, and (3) the perturbation classification + expanded suites are
present (LIBERO-Plus expands each suite into thousands of perturbation instances). Does not
render (that needs a GPU device, exercised by demo_sim_sanity.sh at run time). Exits
non-zero on any failure so `ryzers run` / CI catches a broken image early.
"""
import sys


def main() -> int:
    import mujoco
    import robosuite  # noqa: F401
    from libero.libero import benchmark  # noqa: F401
    from libero.libero.envs import OffScreenRenderEnv  # noqa: F401

    import sim_liberoplus  # noqa: F401
    from sim_liberoplus.libero_env import (
        PERTURBATION_CATEGORIES,
        SUITES,
        get_benchmark_dict,
        get_max_steps,
        load_task_classification,
    )
    from sim_liberoplus.policy import Policy, load_policy

    policy = load_policy()  # default: built-in RandomPolicy
    if not isinstance(policy, Policy):
        print("FAIL: default policy is not a sim_liberoplus.Policy", file=sys.stderr)
        return 1

    benchmark_dict = get_benchmark_dict()
    for suite in SUITES:
        if suite not in benchmark_dict:
            print(f"FAIL: suite {suite} missing from LIBERO benchmark dict", file=sys.stderr)
            return 1
        get_max_steps(suite)

    # LIBERO-Plus perturbation classification: report per-suite task counts + category spread.
    classification = load_task_classification()
    total = 0
    cats = set()
    per_suite = {}
    for suite in SUITES:
        entries = classification.get(suite, [])
        per_suite[suite] = len(entries)
        total += len(entries)
        cats.update(e.get("category") for e in entries)
    missing = [c for c in PERTURBATION_CATEGORIES if c not in cats]
    if missing:
        print(f"FAIL: classification missing categories: {missing}", file=sys.stderr)
        return 1

    print(f"mujoco           : {mujoco.__version__}")
    print(f"suites           : {', '.join(SUITES)}")
    print(f"tasks/suite      : {per_suite}")
    print(f"total perturbed  : {total} task instances")
    print(f"categories       : {len(cats)} ({', '.join(sorted(cats))})")
    print(f"default policy   : {getattr(policy, 'name', type(policy).__name__)}")
    print("deps import ok   : mujoco, robosuite, libero (envs), sim_liberoplus + RandomPolicy")
    print("PASS: LIBERO-Plus simulator env OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
