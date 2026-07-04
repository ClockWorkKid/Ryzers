# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# RoboTwin 2.0 closed-loop stack (SAPIEN offscreen Vulkan RT + mplib motion
# planning) layered on the fastwam image. curobo (CUDA-only) is replaced by mplib
# and the OIDN denoiser is disabled via patches/robotwin_rocm.patch. RoboTwin sim
# assets are third-party and fetched at runtime by scripts/setup_robotwin.sh.

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      mesa-vulkan-drivers libvulkan1 vulkan-tools \
      libgl1 libglx-mesa0 libegl-mesa0 libgles2 \
      libxrender1 libxext6 libsm6 libxi6 libxrandr2 libxfixes3 libxcursor1 libxinerama1 libglib2.0-0 \
      ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# SAPIEN sim deps under a torch/numpy constraint so the base ROCm torch stays put.
# setuptools is pinned <81 because sapien/mplib import the removed pkg_resources.
RUN pip freeze | grep -iE "^(torch|torchvision|torchcodec)==" > /opt/torch-constraints.txt && \
    PIP_CONSTRAINT=/opt/torch-constraints.txt pip install --no-cache-dir \
      "setuptools<81" \
      "sapien==3.0.0b1" "transforms3d==0.4.2" "trimesh==4.4.3" "gymnasium==0.29.1" \
      toppra scipy "pyglet<2" termcolor moviepy av matplotlib pydantic zarr h5py lxml && \
    PIP_CONSTRAINT=/opt/torch-constraints.txt pip install --no-cache-dir --no-deps "mplib==0.2.1" && \
    pip install --no-cache-dir "numpy==1.26.4"

# Apply the ROCm compatibility patch to the FastWAM-vendored RoboTwin (curobo->mplib,
# denoiser off, optional open3d). This diff is the only change to the vendored tree.
COPY patches/robotwin_rocm.patch /ryzers/patches/robotwin_rocm.patch
RUN git -C /repos/fastwam apply --whitespace=nowarn /ryzers/patches/robotwin_rocm.patch && \
    echo "applied robotwin_rocm.patch"

RUN python -c "import torch, sapien, mplib.planner; from mplib.sapien_utils import SapienPlanner; assert torch.version.hip, torch.__version__; print('OK torch', torch.__version__, 'sapien', sapien.__version__)"

COPY test.py /ryzers/
COPY scripts/ /ryzers/scripts/
COPY demos/ /ryzers/demos/
RUN chmod +x /ryzers/scripts/*.sh /ryzers/demos/*.sh

CMD ["python", "/ryzers/test.py"]
