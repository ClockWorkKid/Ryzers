#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Capability 2: open-loop replay of a real LIBERO demo episode. Feeds the recorded
# observations to VLA-JEPA's predict_action and overlays predicted vs ground-truth
# 7-DoF actions per dimension (rule 2.a). First run downloads one demo HDF5.
#
#   ryzers run /ryzers/demo_openloop.sh
#   SUITE=libero_object TASK_ID=0 EPISODE=0 ryzers run /ryzers/demo_openloop.sh
#   GT_LOCAL=/models/my_demo.hdf5 ryzers run /ryzers/demo_openloop.sh
set -euo pipefail
exec python /ryzers/openloop_replay.py
