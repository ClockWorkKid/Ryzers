# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# ImageWAM (FLUX.2 [klein] image-editing world-action model) policy/model layer for AMD
# Strix Halo (gfx1151). ImageWAM's thesis is that a single image-editing step (predict the
# future/goal frame) replaces video generation, so the "dream" is one edited image, not a
# clip. This is a slim policy image: it installs only ImageWAM on the base image's ROCm torch
# (upstream CUDA cu118 torch pins are stripped so the base ROCm torch is preserved). It ships
# NO simulator. It composes on:
#   * the plain ROCm base  -> non-sim demos (smoke / weights / open-loop / dream)
#   * simulation/libero    -> closed-loop + interactive LIBERO  (ryzers build libero imagewam)
#   * simulation/robotwin  -> closed-loop RoboTwin (via experiments/robotwin/imagewam_policy)
# Installing under a constraint pinning the base's torch+numpy lets the exact same layer
# compose on a plain base (numpy 2.x) or on a sim base (numpy 1.26.4) without conflict.
# FLUX.2 base + AE weights, Qwen3 text encoder and the ImageWAM checkpoint are fetched at
# runtime by scripts/ (gated repos need HF_TOKEN); they are never re-hosted here.

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# Upstream pins (see docs/UPSTREAM_PIN.commit.txt).
ARG IMAGEWAM_COMMIT=70bc47147a49652fc3ca62617cab0e0b6b31bc92
ARG FLUX2_COMMIT=50fe5162777813d869182b139e83b10743caef15

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    FLUX2_SRC=/repos/flux2

WORKDIR /ryzers

RUN apt-get update && \
    apt-get install -y --no-install-recommends git cmake build-essential && \
    rm -rf /var/lib/apt/lists/*

# Upstream ImageWAM + the FLUX.2 source it builds on (FLUX.2 is used via PYTHONPATH, not
# pip-installed, per upstream). Both cloned + pinned at build time (compliance rule 8).
RUN git clone https://github.com/yuyangalin/ImageWAM /repos/imagewam && \
    git -C /repos/imagewam checkout ${IMAGEWAM_COMMIT} && \
    git clone https://github.com/black-forest-labs/flux2 /repos/flux2 && \
    git -C /repos/flux2 checkout ${FLUX2_COMMIT}

# Pin the base's torch + numpy so the ImageWAM install (cu118 pins stripped) never replaces
# them. This is what lets one layer compose on the plain ROCm base (numpy 2.x) and on a
# simulator base (numpy 1.26.4) identically. Mirrors the fastwam approach.
COPY scripts/strip_cuda_torch.py /ryzers/scripts/strip_cuda_torch.py
RUN python /ryzers/scripts/strip_cuda_torch.py /repos/imagewam/pyproject.toml && \
    pip freeze | grep -iE "^(torch|torchvision|torchcodec|numpy)==" > /opt/imagewam-constraints.txt && \
    cat /opt/imagewam-constraints.txt

# Install ImageWAM + its shared (eval) extra under the torch/numpy constraint. FLUX.2 needs
# transformers==4.56.1 (upstream base pins 4.51.3 for OmniGen2/Ovis), so bump it after. The
# CUDA-leaning training deps (deepspeed) install import-only here; bitsandbytes is not needed
# for FLUX.2 inference. If any ROCm-incompatible dep breaks the build we patch it (rule 2.1).
RUN PIP_CONSTRAINT=/opt/imagewam-constraints.txt pip install --no-cache-dir -e "/repos/imagewam[shared]" && \
    PIP_CONSTRAINT=/opt/imagewam-constraints.txt pip install --no-cache-dir "transformers==4.56.1" && \
    PIP_CONSTRAINT=/opt/imagewam-constraints.txt pip install --no-cache-dir \
      "hf-transfer>=0.1.8" "sentencepiece>=0.2" "pyarrow>=17" "av>=12" \
      "imageio>=2.34" "imageio-ffmpeg>=0.5" "matplotlib>=3.8" "opencv-python-headless"

# Sim Policy adapters (sim_libero.Policy / sim_robotwin.Policy wrapping the ImageWAM model).
# Only imported when driving a sim base via POLICY_FACTORY=imagewam_{libero,robotwin}_policy:
# build_policy; harmless (unused) on the plain base.
COPY adapters/ /opt/imagewam-adapters/

# Import sign-of-life at build: ROCm torch + ImageWAM package + FLUX.2 runtime factory import.
RUN python -c "import torch; assert torch.version.hip, 'torch is not a ROCm build: '+torch.__version__; import imagewam; from imagewam.runtime import create_imagewam_flux2_klein; print('OK imagewam on torch', torch.__version__, 'hip', torch.version.hip)"

COPY test.py /ryzers/
COPY scripts/ /ryzers/scripts/
COPY demos/ /ryzers/demos/
RUN chmod +x /ryzers/scripts/*.sh /ryzers/demos/*.sh 2>/dev/null || true

CMD ["python", "/ryzers/test.py"]
