#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Single-token PYTHON shim. The upstream start_server.sh launches the server with a QUOTED
# "${PYTHON}", so PYTHON must resolve to one executable token. This shim lets the demo set
# PYTHON=/ryzers/scripts/opt_python.sh to route the server through opt_launch.py (which arms the
# gfx1151 default-route speedups) while passing the server script + all args through verbatim.
here="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
exec python "${here}/opt_launch.py" "$@"
