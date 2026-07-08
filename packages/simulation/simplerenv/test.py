# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Import sign-of-life for the SimplerEnv sim base (no GPU render). Verifies the SAPIEN +
ManiSkill2-real2sim + SimplerEnv + harness stack imports and lists the available tasks."""
import sys


def main() -> int:
    import sapien.core as sapien
    import mani_skill2_real2sim  # noqa: F401
    import simpler_env  # noqa: F401

    import sim_simplerenv
    from sim_simplerenv.policy import load_policy
    from sim_simplerenv import simplerenv_env as se

    print(f"sapien            : {sapien.__version__}")
    print(f"google tasks      : {se.GOOGLE_TASKS}")
    print(f"widowx tasks      : {se.WIDOWX_TASKS}")
    policy = load_policy()  # default RandomPolicy
    print(f"default policy    : {type(policy).__name__}  (POLICY_FACTORY seam OK)")
    print("PASS: sim_simplerenv import sign-of-life OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
