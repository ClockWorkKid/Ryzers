# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Model-agnostic RoboCasa simulator base image for AMD Strix Halo (gfx1151).
# Ships the RoboCasa / robosuite / MuJoCo closed-loop kitchen benchmark (headless EGL)
# plus the sim_robocasa harness (Policy seam + closed-loop/interactive/sanity runners and
# a built-in RandomPolicy). No policy/model code lives here: any policy chains on top of
# this image (ryzers build simulation/robocasa <policy>) and is selected at runtime via
# POLICY_FACTORY. robosuite + robocasa are cloned from upstream at pinned commits (code
# only); the ~10GB kitchen assets are third-party and fetched at RUNTIME into a mounted
# volume by scripts/setup_robocasa.sh, never baked here (rules 3 & 8).

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

# RoboCasa (RoboSuite v1.5 backend) closed-loop sim. numpy is pinned to 1.26.4 (matches the
# other sim bases + the xwam layer, so the chained image has one numpy ABI); the constraint
# then protects the base ROCm torch + numpy so they are never replaced by transitive deps.
# robosuite MUST be the master branch (RoboCasa v0.2 requirement). robocasa is installed with
# --no-deps because its setup.py hard-pins numpy==1.23.3 (an overly strict pin that would
# force a numpy downgrade / ABI clash); its real runtime deps are added explicitly and
# mujoco is pinned to 3.2.6 (robocasa asserts exactly this at import). robocasa/__init__.py
# also asserts numpy in {1.23.2,3,5} -- impossible on this Python 3.12 base (numpy>=1.26
# only), so the allow-list is widened to 1.26.4 with a one-line sed (project-compat patch,
# rule 2.1). Both repos are installed editable so their packaged assets/configs resolve at
# their clone paths.
ARG ROBOSUITE_COMMIT=232ce7d4a6ed89c949a9aba024a05c8c32fdd08b
ARG ROBOCASA_COMMIT=756598a5be52e052339bb2d957426e39015c2afb
ENV PYTHONPATH=/opt/sim
RUN pip install --no-cache-dir "numpy==1.26.4" && \
    pip freeze | grep -iE "^(torch|torchvision|torchcodec|numpy)==" > /opt/torch-constraints.txt && \
    git clone https://github.com/ARISE-Initiative/robosuite /opt/robosuite && \
    git -C /opt/robosuite checkout ${ROBOSUITE_COMMIT} && \
    PIP_CONSTRAINT=/opt/torch-constraints.txt pip install --no-cache-dir -e /opt/robosuite && \
    git clone https://github.com/robocasa/robocasa /opt/robocasa && \
    git -C /opt/robocasa checkout ${ROBOCASA_COMMIT} && \
    sed -i 's#\("1.23.5",\)#\1\n    "1.26.4",#' /opt/robocasa/robocasa/__init__.py && \
    PIP_CONSTRAINT=/opt/torch-constraints.txt pip install --no-cache-dir --no-deps -e /opt/robocasa && \
    PIP_CONSTRAINT=/opt/torch-constraints.txt pip install --no-cache-dir \
      "mujoco==3.2.6" numba scipy pygame pyyaml pynput tqdm termcolor h5py lxml easydict \
      "opencv-python-headless" "imageio>=2.34" "imageio-ffmpeg>=0.5" "pillow>=10" && \
    pip install --no-cache-dir "numpy==1.26.4"

# Set up robocasa system macros (non-interactive) and pre-create the kitchen-assets dir as
# a symlink target so the runtime download (scripts/setup_robocasa.sh) lands on the mounted
# volume instead of inside the image layer. The macros file is generated in the package.
RUN python -m robocasa.scripts.setup_macros && \
    ROBOCASA_PKG=$(python -c "import importlib.util, os; print(os.path.dirname(importlib.util.find_spec('robocasa').origin))") && \
    echo "robocasa package: $ROBOCASA_PKG"

# The sim harness (Policy seam, closed-loop/interactive/sanity runners, RandomPolicy).
COPY lib/ /opt/sim/

# Verify robosuite + robocasa + the sim_robocasa harness resolve (no env build / GPU here;
# rendering + asset use are exercised at run time by demo_sim_sanity.sh after setup).
RUN python -c "import mujoco, robosuite, robocasa; from robosuite.controllers import load_composite_controller_config; import sim_robocasa; from sim_robocasa.policy import load_policy; from sim_robocasa.robocasa_env import TASKS; print('OK mujoco', mujoco.__version__, '| robosuite', robosuite.__version__, '|', len(TASKS), 'robocasa tasks | sim_robocasa imports cleanly')"

COPY test.py /ryzers/
COPY scripts/ /ryzers/scripts/
COPY demos/ /ryzers/demos/
RUN chmod +x /ryzers/scripts/*.sh /ryzers/demos/*.sh

CMD ["python", "/ryzers/test.py"]
