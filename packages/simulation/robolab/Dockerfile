# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Model-agnostic RoboLab-AMD simulator base image for AMD Strix Halo (gfx1151).
# AMD-native reimplementation of NVIDIA RoboLab-120 task semantics on the open robosuite/
# MuJoCo backend (headless EGL) plus the sim_robolab harness (Policy seam + sanity/
# interactive runners + a built-in joint-space RandomPolicy). No policy/model code lives
# here: a policy chains on top (ryzers build simulation/robolab <policy>) and is selected at
# runtime via POLICY_FACTORY. robosuite is installed from PyPI at a pinned version (code
# only). The pilot task (BananaInBowl) uses the REAL YCB 011_banana + 024_bowl meshes (the
# same source assets upstream RoboLab uses); they are fetched from the open YCB S3 bucket and
# convex-decomposed (coacd) at RUNTIME into the mounted assets volume by scripts/setup_robolab.sh
# (or lazily on first env build) -- never baked (rules 3 & 8). A primitive fallback keeps the
# sim running if the fetch is unavailable.

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /ryzers

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      git cmake build-essential curl \
      libegl-dev libgles-dev libosmesa6-dev libglew-dev libglu1-mesa libglu1-mesa-dev \
      libglib2.0-0 ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# robosuite v1.5.1 (the API the custom BananaInBowl env is written against: Robotiq85Gripper,
# JOINT_POSITION composite controller, ManipulationEnv). numpy is pinned to 1.26.4 to match the
# other sim bases + the ROCm torch ABI; the constraint then protects the base ROCm torch +
# numpy so they are never replaced by transitive deps. mujoco is pinned to a robosuite-1.5
# compatible release.
ARG ROBOSUITE_VERSION=1.5.1
ENV PYTHONPATH=/opt/sim
RUN pip install --no-cache-dir "numpy==1.26.4" && \
    pip freeze | grep -iE "^(torch|torchvision|torchcodec|numpy)==" > /opt/torch-constraints.txt && \
    PIP_CONSTRAINT=/opt/torch-constraints.txt pip install --no-cache-dir \
      "robosuite==${ROBOSUITE_VERSION}" && \
    PIP_CONSTRAINT=/opt/torch-constraints.txt pip install --no-cache-dir \
      "mujoco==3.2.6" numba scipy pyyaml tqdm termcolor h5py lxml easydict \
      "opencv-python-headless" "imageio>=2.34" "imageio-ffmpeg>=0.5" "pillow>=10" \
      "trimesh>=4.0" "coacd>=1.0" "matplotlib>=3.7" && \
    pip install --no-cache-dir "numpy==1.26.4"

# Non-interactive robosuite macros (silences the private-macro warning).
RUN python -m robosuite.scripts.setup_macros 2>/dev/null || true

# The sim harness (Policy seam, sanity/interactive runners, RandomPolicy, custom env).
COPY lib/ /opt/sim/

# Verify robosuite + the sim_robolab harness + the custom env resolve (no env build / GPU
# here; rendering + control are exercised at run time by demo_sim_sanity.sh).
RUN python -c "import mujoco, robosuite; from robosuite.controllers import load_composite_controller_config; import sim_robolab; from sim_robolab.policy import load_policy; from sim_robolab.robolab_env import TASKS, build_controller_config; from sim_robolab.tasks.banana_in_bowl import BananaInBowl; print('OK mujoco', mujoco.__version__, '| robosuite', robosuite.__version__, '|', len(TASKS), 'robolab tasks | sim_robolab imports cleanly')"

COPY test.py /ryzers/
COPY scripts/ /ryzers/scripts/
COPY demos/ /ryzers/demos/
RUN chmod +x /ryzers/scripts/*.sh /ryzers/demos/*.sh

CMD ["python", "/ryzers/test.py"]
