# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# AVDC (flow-diffusion) - "Learning to Act from Actionless Videos through Dense Correspondences"
# - a text-conditioned 3D-UNet video-diffusion policy, for AMD Strix Halo (gfx1151). Installs the
# AVDC model path on the base image's ROCm torch. Upstream targets CUDA torch 1.x + Python 3.9
# (conda); we drop those pins, keep the base ROCm torch, and install only the model-path deps AVDC
# imports (rule 2.1 - rely on upstream, patch only what ROCm needs).
#
# Model path (this image, phases 0-2 - forward smoke + open-loop video prediction):
#  * 3D UNet video-diffusion (flowdiffusion + vendored guided-diffusion UNetModel, dims=3,
#    task-token cross-attention) - Meta-World UnetMW (128x128, 7 future frames).
#  * OpenAI CLIP ViT-B/32 text encoder (transformers CLIPTextModel) -> 512-d task tokens.
#  * GoalGaussianDiffusion sampler (imagen-pytorch style, pred_v, DDIM).
# All attention is plain matmul/softmax -> nothing ROCm-specific to patch in the forward math.
#
# Sim path (phase 3 - closed-loop Meta-World) is added as its own layers later (single self-
# contained image, rule 0.4): vendored Meta-World (mujoco-py) + UniMatch optical flow. Weights
# (CLIP + AVDC snapshots) are fetched at runtime (rule 8).

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG AVDC_COMMIT=e9b5b0098e59da0e1fe24a18ed9bec5b825e2ca2

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /ryzers

RUN apt-get update && \
    apt-get install -y --no-install-recommends git cmake build-essential ffmpeg && \
    rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/flow-diffusion/AVDC_experiments /repos/avdc && \
    git -C /repos/avdc checkout ${AVDC_COMMIT}

# Pin the base's torch + torchvision + numpy so nothing pulls a CUDA build or breaks the torch
# ABI, then install AVDC's model-path deps under that constraint. transformers 4.34.0 + accelerate
# 0.23.0 + ema_pytorch 0.2.3 match the upstream requirements (the Trainer/EMA/CLIP inference API);
# flowdiffusion (with its vendored guided_diffusion sub-package) is installed editable, no-deps.
RUN pip freeze | grep -iE "^(torch|torchvision|numpy)==" > /opt/avdc-constraints.txt && \
    PIP_CONSTRAINT=/opt/avdc-constraints.txt pip install --no-cache-dir \
      "transformers==4.34.0" \
      "accelerate==0.23.0" \
      "ema_pytorch==0.2.3" \
      "einops==0.7.0" "einops_exts==0.0.4" \
      "pytorch_fid==0.3.0" "tensorboard" \
      "matplotlib" "imageio[ffmpeg]" "pynvml" "tqdm" \
      "Pillow" "opencv-python-headless" "safetensors" && \
    PIP_CONSTRAINT=/opt/avdc-constraints.txt pip install --no-cache-dir --no-deps -e /repos/avdc/flowdiffusion

# Fail early if the torch stack was clobbered to a non-ROCm build by the deps above.
RUN python -c "import torch; assert torch.version.hip, 'torch is not a ROCm build: '+torch.__version__; print('OK torch', torch.__version__, 'hip', torch.version.hip)"

# ---------------------------------------------------------------------------------------------
# Meta-World closed-loop simulator layer (phase 3) - single self-contained image (rule 0.4).
# The AVDC native benchmark is Meta-World, whose vendored fork uses mujoco-py (MuJoCo 2.1) + gym.
# On gfx1151 mujoco-py runs headless via EGL on the iGPU (physics on CPU, offscreen GL on GPU).
# ---------------------------------------------------------------------------------------------
# Only add mujoco210's lib dir; ROCm + system GL libs resolve via ldconfig. Do NOT prepend
# /usr/lib/x86_64-linux-gnu here - it shadows ROCm's libhsa-runtime64 and breaks the torch import.
ENV MUJOCO_GL=egl \
    MUJOCO_PY_MUJOCO_PATH=/root/.mujoco/mujoco210 \
    LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      libgl1-mesa-dev libglew-dev libosmesa6-dev libglfw3 patchelf wget && \
    rm -rf /var/lib/apt/lists/*

# MuJoCo 2.1 runtime (fetched from the upstream DeepMind release at build - rule 8, not re-hosted).
RUN mkdir -p /root/.mujoco && cd /root/.mujoco && \
    wget -q https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz && \
    tar -xzf mujoco210-linux-x86_64.tar.gz && rm mujoco210-linux-x86_64.tar.gz

# The Meta-World / mujoco-py / gym 0.25 stack predates numpy 2; mujoco-py's compiled read_pixels
# overflows on numpy>=2 segmentation buffers (OverflowError: 256 out of bounds for uint8). Pin
# numpy<2 image-wide (torch 2.10 + transformers + flowdiffusion all run fine on 1.26). cython<3 is
# needed to cythonize mujoco-py on py3.12; gym 0.25.2 keeps the classic API Meta-World expects.
RUN pip install --no-cache-dir "numpy==1.26.4" "cython<3" "mujoco-py<2.2" "gym==0.25.2" "scipy" && \
    pip install --no-cache-dir --no-deps -e /repos/avdc/metaworld

# Pre-cythonize/compile mujoco-py's EGL backend at build (first import compiles cymj) so runtime
# start-up is fast and any build break is caught here rather than at first rollout.
RUN python -c "import mujoco_py; print('OK mujoco_py', mujoco_py.__version__)"

# UniMatch (GMFlow) dense-flow weights for the video->flow->action pipeline, fetched from the
# upstream unimatch S3 at build (rule 8). get_flow_model() reads pretrained/ relative to experiment/.
RUN mkdir -p /repos/avdc/experiment/pretrained && \
    wget -q -O /repos/avdc/experiment/pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth \
      https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth

# Minimal ROCm/API source patches on top of the pinned upstream (rule 2.1). Applied after the pip
# layers (patches only touch the git checkout) so patch edits don't invalidate deps caching.
COPY patches/ /tmp/avdc-patches/
RUN python /tmp/avdc-patches/rocm_port.py /repos/avdc

# flowdiffusion (model) + experiment/ (closed-loop policy, UniMatch, env wrappers, name2maskid)
# + /ryzers (our avdc_optim shim).
ENV PYTHONPATH=/repos/avdc/flowdiffusion:/repos/avdc/experiment:/ryzers

COPY test.py /ryzers/
COPY avdc_optim.py /ryzers/
COPY scripts/ /ryzers/scripts/
COPY demos/ /ryzers/demos/
RUN chmod +x /ryzers/scripts/*.sh /ryzers/demos/*.sh 2>/dev/null || true

CMD ["python", "/ryzers/test.py"]
