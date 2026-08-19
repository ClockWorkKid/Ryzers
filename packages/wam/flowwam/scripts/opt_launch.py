# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Thin launcher that applies the default-route FlowWAM optimizations (flowwam_opt.patch_class)
and then runs a target script or module UNCHANGED, so the upstream flow-action server stays
untouched.

    python /ryzers/scripts/opt_launch.py <script.py> [args...]
    python /ryzers/scripts/opt_launch.py -m <module>   [args...]

Wired into demo_closedloop_robotwin.sh by exporting PYTHON="python .../opt_launch.py", which the
upstream start_server.sh honors (PYTHON="${PYTHON:-python}") to launch flow_action_server.py.

patch_class() is a no-op unless `diffsynth` is importable in this process (PYTHONPATH already set
by the demo) and can be disabled with FLOWWAM_OPT=0. The target then sees exactly the argv it
would have seen without the wrapper.
"""
import os
import runpy
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # make flowwam_opt importable

try:
    import flowwam_opt
    flowwam_opt.patch_class()
except Exception as e:  # noqa: BLE001 - never let the optimizer block the demo
    print(f"[flowwam_opt] optimizations NOT applied ({type(e).__name__}: {e})", file=sys.stderr)

_argv = sys.argv[1:]
if not _argv:
    print("usage: opt_launch.py <script.py|-m module> [args...]", file=sys.stderr)
    raise SystemExit(2)

if _argv[0] == "-m":
    if len(_argv) < 2:
        print("usage: opt_launch.py -m <module> [args...]", file=sys.stderr)
        raise SystemExit(2)
    _module = _argv[1]
    sys.argv = [_module] + _argv[2:]
    runpy.run_module(_module, run_name="__main__", alter_sys=True)
else:
    _script = _argv[0]
    sys.argv = [_script] + _argv[1:]
    runpy.run_path(_script, run_name="__main__")
