# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# FastWAM (Wan2.2-TI2V-5B world-action model) on AMD Strix Halo (gfx1151).
# Direct PyTorch port on the base image's ROCm torch (torch 2.10.0+rocm7.2.2);
# upstream's CUDA torch pins are stripped so the base ROCm torch is preserved.
# Adds the LIBERO/MuJoCo closed-loop stack (headless EGL). Weights, datasets and
# LIBERO assets are fetched at runtime by scripts/, never baked into the image.

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG FASTWAM_COMMIT=45d8e1458921d83f8ad6cf9ce993d371208dabd0

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /ryzers

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      git cmake build-essential \
      libegl-dev libgles-dev libosmesa6-dev libglew-dev libglu1-mesa libglu1-mesa-dev && \
    rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/yuantianyuan01/FastWAM /repos/fastwam && \
    git -C /repos/fastwam checkout ${FASTWAM_COMMIT}

COPY scripts/strip_cuda_torch.py /ryzers/scripts/strip_cuda_torch.py
RUN python /ryzers/scripts/strip_cuda_torch.py /repos/fastwam/pyproject.toml && \
    pip install --no-cache-dir -e /repos/fastwam && \
    pip install --no-cache-dir "hf-transfer>=0.1.8" "sentencepiece>=0.2" \
      "pyarrow>=17" "av>=12" "imageio-ffmpeg>=0.5" "matplotlib>=3.8" "opencv-python-headless"

# LIBERO closed-loop sim. numpy is pinned to 1.26.4 (robosuite's numba has no
# py3.12 wheel for numpy 2.x); the constraint then protects torch + numpy so the
# base ROCm torch is never replaced. LIBERO installs with --no-deps for the same reason.
# The `libero` package ships no top-level __init__.py, so it resolves via PYTHONPATH.
ENV PYTHONPATH=/opt/LIBERO
RUN pip install --no-cache-dir "numpy==1.26.4" && \
    pip freeze | grep -iE "^(torch|torchvision|torchcodec|numpy)==" > /opt/torch-constraints.txt && \
    PIP_CONSTRAINT=/opt/torch-constraints.txt pip install --no-cache-dir \
      "robosuite==1.4.0" "bddl==1.0.1" "mujoco==3.3.2" easydict thop && \
    git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO /opt/LIBERO && \
    PIP_CONSTRAINT=/opt/torch-constraints.txt pip install --no-cache-dir --no-deps -e /opt/LIBERO

# First `import libero.libero` writes ~/.libero/config.yaml interactively; feed
# defaults so it is baked into the image and never prompts at runtime.
RUN printf 'N\nN\nN\nN\nN\n' | python -c "import torch, mujoco, robosuite; from libero.libero import benchmark; assert torch.version.hip, 'torch is not a ROCm build: '+torch.__version__; print('OK torch', torch.__version__, 'hip', torch.version.hip, 'mujoco', mujoco.__version__)"

COPY test.py /ryzers/
COPY scripts/ /ryzers/scripts/
COPY demos/ /ryzers/demos/
RUN chmod +x /ryzers/scripts/*.sh /ryzers/demos/*.sh

CMD ["python", "/ryzers/test.py"]
