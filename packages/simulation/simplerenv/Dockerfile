# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Model-agnostic SimplerEnv (real-to-sim) simulator base for AMD Strix Halo (gfx1151).
# SimplerEnv evaluates real-world manipulation policies in simulation on two embodiments
# (Google Robot / WidowX+Bridge) via a Gym API. We target SimplerEnv *main* = ManiSkill2 +
# SAPIEN 2 on purpose:
#   * ManiSkill3's speed comes from GPU-parallel PhysX, which is CUDA/NVIDIA-only and does
#     not accelerate on gfx1151; ManiSkill2 uses CPU physics (portable) + SAPIEN offscreen
#     Vulkan rendering (AMD-friendly, same Mesa RADV path proven by simulation/robotwin).
#   * The visual-matching ManiSkill2 setup is what the VLA-JEPA paper's SimplerEnv numbers
#     use, so it is the faithful reproduction target.
# No policy/model code lives here: a policy chains on top (ryzers build simulation/simplerenv
# <policy>) and plugs in via the sim_simplerenv.Policy seam (interactive) and/or SimplerEnv's
# own inference loop (closed-loop parity). Real-to-sim assets are third-party and fetched at
# runtime by scripts/setup_simplerenv.sh, never baked here (compliance).
#
# NOTE (first-pass, to validate on hardware): the exact sapien 2.x wheel availability for
# py3.12 and ManiSkill2_real2sim's transitive pins may need adjustment; keep upstream code
# and only patch for ROCm/gfx bugs (see patches/ if added).

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /ryzers

# Vulkan (offscreen) + GL/X libs SAPIEN needs headless, plus ffmpeg for video I/O.
# Mirror of simulation/robotwin's system deps (proven SAPIEN Vulkan stack on gfx1151).
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      git wget \
      mesa-vulkan-drivers libvulkan1 vulkan-tools \
      libgl1 libglx-mesa0 libegl-mesa0 libgles2 \
      libxrender1 libxext6 libsm6 libxi6 libxrandr2 libxfixes3 libxcursor1 libxinerama1 libglib2.0-0 \
      ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# SimplerEnv (ManiSkill2 real2sim) under a torch/numpy constraint so the base ROCm torch
# stays put. setuptools<81 because sapien imports the removed pkg_resources. numpy pinned
# last (SimplerEnv wants 1.24.x). sapien 2.x provides the CPU-physics + Vulkan-render engine.
ARG SIMPLERENV_COMMIT=main
RUN pip freeze | grep -iE "^(torch|torchvision|torchcodec)==" > /opt/torch-constraints.txt && \
    PIP_CONSTRAINT=/opt/torch-constraints.txt pip install --no-cache-dir \
      "setuptools<81" "sapien==2.2.2" "transforms3d==0.4.2" "trimesh" "gymnasium==0.29.1" \
      "mplib" "pyyaml" scipy "pyglet<2" termcolor moviepy av matplotlib \
      "imageio>=2.34" "imageio-ffmpeg>=0.5" "pillow>=10" opencv-python-headless \
      huggingface_hub && \
    git clone --recurse-submodules https://github.com/simpler-env/SimplerEnv /opt/SimplerEnv && \
    if [ "${SIMPLERENV_COMMIT}" != "main" ]; then git -C /opt/SimplerEnv checkout ${SIMPLERENV_COMMIT}; fi && \
    PIP_CONSTRAINT=/opt/torch-constraints.txt pip install --no-cache-dir -e /opt/SimplerEnv/ManiSkill2_real2sim && \
    PIP_CONSTRAINT=/opt/torch-constraints.txt pip install --no-cache-dir -e /opt/SimplerEnv && \
    pip install --no-cache-dir "numpy==1.24.4" && \
    rm -rf /opt/SimplerEnv/.git

# The sim harness (Policy seam, env loader, interactive/sanity runners, random policy).
ENV PYTHONPATH=/opt/sim:/opt/SimplerEnv
COPY lib/ /opt/sim/

# Import sign-of-life (no GPU render at build time).
RUN python -c "import sapien.core as sapien; import mani_skill2_real2sim; import simpler_env; import sim_simplerenv; from sim_simplerenv.policy import load_policy; print('OK sapien', sapien.__version__, '| simpler_env + sim_simplerenv import cleanly')"

COPY test.py /ryzers/
COPY scripts/ /ryzers/scripts/
COPY demos/ /ryzers/demos/
RUN chmod +x /ryzers/scripts/*.sh /ryzers/demos/*.sh

CMD ["python", "/ryzers/test.py"]
