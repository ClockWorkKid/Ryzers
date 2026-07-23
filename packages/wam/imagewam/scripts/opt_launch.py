# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Thin launcher that applies the default-route ImageWAM optimizations (imagewam_opt.patch_class)
and then runs a target script or module UNCHANGED, so upstream evaluators and the interactive
servers stay untouched (rules 2.1 / 0.0).

    python /ryzers/scripts/opt_launch.py <script.py> [args...]
    python /ryzers/scripts/opt_launch.py -m <module>   [args...]

patch_class() is a no-op unless `imagewam` is importable in this process (PYTHONPATH already set
by the demo) and can be disabled with IMAGEWAM_OPT=0. The target then sees exactly the argv it
would have seen without the wrapper.
"""
import os
import runpy
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # make imagewam_opt importable

try:
    import imagewam_opt
    imagewam_opt.patch_class()
except Exception as e:  # noqa: BLE001 - never let the optimizer block the demo
    print(f"[imagewam_opt] optimizations NOT applied ({type(e).__name__}: {e})", file=sys.stderr)

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
