# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Environment sign-of-life for the RoboLab-AMD simulator base image (Strix Halo, gfx1151).

Runs inside the built image with NO model. Proves the container has (1) the robosuite /
MuJoCo stack importing cleanly, (2) the custom BananaInBowl env + JOINT_POSITION controller
config resolving, and (3) the sim_robolab harness + built-in RandomPolicy resolving, before
a policy image is chained on top. Does not build an env or render (that needs a GPU device,
exercised by demo_sim_sanity.sh at run time). Exits non-zero on any failure.
"""
import sys


def main() -> int:
    import mujoco
    import robosuite  # noqa: F401
    from robosuite.controllers import load_composite_controller_config  # noqa: F401

    import sim_robolab  # noqa: F401
    from sim_robolab.policy import Policy, load_policy
    from sim_robolab.robolab_env import TASKS, build_controller_config, get_max_steps
    from sim_robolab.tasks.banana_in_bowl import BananaInBowl  # noqa: F401

    policy = load_policy()  # default: built-in RandomPolicy
    if not isinstance(policy, Policy):
        print("FAIL: default policy is not a sim_robolab.Policy", file=sys.stderr)
        return 1

    cfg = build_controller_config(robot="Panda")
    if cfg["body_parts"]["right"]["type"] != "JOINT_POSITION":
        print("FAIL: arm controller is not JOINT_POSITION", file=sys.stderr)
        return 1
    if "gripper" not in cfg["body_parts"]["right"]:
        print("FAIL: gripper sub-config missing from arm controller", file=sys.stderr)
        return 1
    for t in TASKS:
        get_max_steps(t)

    print(f"mujoco           : {mujoco.__version__}")
    print(f"robosuite        : {robosuite.__version__}")
    print(f"tasks            : {len(TASKS)} RoboLab-AMD task(s): {', '.join(TASKS)}")
    print(f"controller       : JOINT_POSITION composite (gripper sub-config OK)")
    print(f"default policy   : {getattr(policy, 'name', type(policy).__name__)}")
    print("deps import ok   : mujoco, robosuite, sim_robolab + RandomPolicy + BananaInBowl")
    print("PASS: RoboLab-AMD simulator env OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
