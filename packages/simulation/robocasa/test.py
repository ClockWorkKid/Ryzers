# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Environment sign-of-life for the RoboCasa simulator base image (Strix Halo, gfx1151).

Runs inside the built image with NO model. Proves the container has (1) the RoboCasa /
robosuite / MuJoCo stack importing cleanly and (2) the sim_robocasa harness + built-in
RandomPolicy resolving, before a policy image is chained on top. Does not build an env or
render (that needs kitchen assets + a GPU device, exercised by demo_sim_sanity.sh at run
time). Exits non-zero on any failure so `ryzers run` / CI catches a broken image early.
"""
import sys


def main() -> int:
    import mujoco
    import robocasa  # noqa: F401
    import robosuite  # noqa: F401
    from robosuite.controllers import load_composite_controller_config  # noqa: F401

    import sim_robocasa  # noqa: F401
    from sim_robocasa.policy import Policy, load_policy
    from sim_robocasa.robocasa_env import TASKS, get_max_steps

    policy = load_policy()  # default: built-in RandomPolicy
    if not isinstance(policy, Policy):
        print("FAIL: default policy is not a sim_robocasa.Policy", file=sys.stderr)
        return 1

    if len(TASKS) != 24:
        print(f"FAIL: expected 24 RoboCasa tasks, got {len(TASKS)}", file=sys.stderr)
        return 1
    for t in TASKS:
        get_max_steps(t)

    print(f"mujoco           : {mujoco.__version__}")
    print(f"robosuite        : {robosuite.__version__}")
    print(f"tasks            : {len(TASKS)} kitchen tasks (e.g. {', '.join(TASKS[:4])} ...)")
    print(f"default policy   : {getattr(policy, 'name', type(policy).__name__)}")
    print("deps import ok   : mujoco, robosuite, robocasa, sim_robocasa + RandomPolicy")
    print("PASS: RoboCasa simulator env OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
