# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# GR-1 (ByteDance) — "Unleashing Large-Scale Video Generative Pre-training for Visual Robot
# Manipulation" — GPT-style vision-language-action policy for AMD Strix Halo (gfx1151).
# Installs GR-1's upstream code on the base image's ROCm torch. Upstream pins torch 1.12.1 +
# CUDA (install.sh); we drop those pins, keep the base ROCm torch, and install only the model-
# path deps GR-1 imports (rule 2.1 — rely on upstream, patch only what ROCm needs).
#
# Model path (this image, phases 0-2 — forward smoke + open-loop prediction):
#  * MAE ViT-B/16 image encoder (models/vision_transformer.py, in-repo, timm-style attention)
#  * OpenAI CLIP ViT-B/32 text encoder (clip.load / clip.tokenize)
#  * Perceiver resampler (flamingo_pytorch) compressing 196 patch tokens -> 9 latents
#  * GPT-2 trajectory backbone (models/trajectory_gpt2.py, vendored) via transformers GPT2Config
#  * action head (arm 6-DoF + gripper) and MAE-style future-frame decoder heads
# All attention is plain matmul+softmax (no flash/xformers) -> nothing ROCm-specific to patch
# in the forward math.
#
# Sim path (phase 3 — closed-loop eval): the CALVIN PyBullet simulator (mees/calvin: calvin_env +
# calvin_agent + tacto) is added as its own layers on top of the model path (single self-contained
# image, rule 0.4). It renders headless via EGL on the AMD iGPU (Mesa/gfx1151). We install only the
# sim + eval deps GR-1's closed-loop path imports, under the same base torch/numpy constraint so the
# ROCm torch stack is never clobbered; we skip CALVIN's numba (unused), pyhash (avoided by the
# closed-loop runner) and its pytorch-lightning/torch-1.13 training pins. CALVIN pins
# hydra-core==1.1.1 which crashes on Python 3.12 (dataclass mutable-default), so we use 1.3.2.
#
# ROCm source patches (patches/rocm_port.py), minimal + idempotent (rule 2.1):
#  * models/gr1.py: the query attention masks are built with `torch.zeros(...).cuda()`; make
#    them device-agnostic (`device=rgb.device`) so the model runs on any explicit device.
#  * evaluation/calvin_evaluation.py: add `map_location='cpu'` to the policy torch.load so a
#    CUDA-saved checkpoint loads regardless of the visible ROCm device ordinal.
#  * models/trajectory_gpt2.py: the vendored HF-4.5-era GPT-2 decorates GPT2Model with
#    `add_code_sample_docstrings(tokenizer_class=...)`; transformers 4.36.2 renamed that kwarg
#    to `processor_class`, so rename it (docs-only) to import on the base image's transformers.
# Weights (MAE ViT-B, GR-1 snapshots, CLIP ViT-B/32) are fetched at runtime (rule 8).

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG GR1_COMMIT=4a2e956bb497586d82489eb4c98b27fb82b5ff47

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /ryzers

RUN apt-get update && \
    apt-get install -y --no-install-recommends git cmake build-essential ffmpeg && \
    rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/bytedance/GR-1 /repos/gr1 && \
    git -C /repos/gr1 checkout ${GR1_COMMIT}

# Pin the base's torch + torchvision + numpy so nothing pulls a CUDA build or breaks the torch
# ABI, then install GR-1's model-path deps under that constraint. transformers 4.36.2 supplies
# every GPT2Config/GPT2Model API the vendored trajectory_gpt2.py needs (verified on-device);
# no torch-1.12-era transformers pin is required.
RUN pip freeze | grep -iE "^(torch|torchvision|numpy)==" > /opt/gr1-constraints.txt && \
    PIP_CONSTRAINT=/opt/gr1-constraints.txt pip install --no-cache-dir \
      "transformers==4.36.2" \
      "tokenizers==0.15.2" \
      "huggingface_hub==0.20.3" \
      "safetensors" \
      "flamingo-pytorch" \
      "einops" "einops-exts" \
      "ftfy" "regex" "tqdm" \
      "Pillow" "opencv-python-headless" "matplotlib" \
      "imageio" "imageio-ffmpeg" && \
    PIP_CONSTRAINT=/opt/gr1-constraints.txt pip install --no-cache-dir \
      "git+https://github.com/openai/CLIP.git"

# Fail early if the torch stack was clobbered to a non-ROCm build by the deps above.
RUN python -c "import torch; assert torch.version.hip, 'torch is not a ROCm build: '+torch.__version__; print('OK torch', torch.__version__, 'hip', torch.version.hip)"

# ---------------------------------------------------------------------------------------------
# CALVIN simulator layer (phase 3, closed-loop eval). Headless PyBullet + EGL on the AMD iGPU.
# ---------------------------------------------------------------------------------------------
ARG CALVIN_COMMIT=fa03f01f19c65920e18cf37398a9ce859274af76

# System GL/EGL/OSMesa for headless pybullet rendering (Ubuntu 24.04 package names; the classic
# libegl1-mesa was dropped in favour of libegl-mesa0). patchelf is used by pybullet's egl setup.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      libegl1 libegl-mesa0 libgl1 libgl1-mesa-dri libglx-mesa0 \
      libosmesa6 libosmesa6-dev patchelf libglew-dev libglfw3-dev mesa-utils && \
    rm -rf /var/lib/apt/lists/*

RUN git clone --recurse-submodules https://github.com/mees/calvin.git /repos/calvin && \
    git -C /repos/calvin checkout ${CALVIN_COMMIT} && \
    git -C /repos/calvin submodule update --init --recursive

# Sim + closed-loop-eval deps only, under the base torch/numpy constraint. hydra 1.3.2 (not
# CALVIN's 1.1.1, which is broken on py3.12); numba/pyhash/pytorch-lightning intentionally omitted.
RUN PIP_CONSTRAINT=/opt/gr1-constraints.txt pip install --no-cache-dir \
      "pybullet" "gym==0.26.2" "hydra-core==1.3.2" "omegaconf==2.3.0" "hydra-colorlog" \
      "numpy-quaternion" "cloudpickle" "gitpython" "rich" "pandas" "scipy" \
      "opencv-python-headless" "moviepy" "termcolor" "plotly" && \
    PIP_CONSTRAINT=/opt/gr1-constraints.txt pip install --no-deps -e /repos/calvin/calvin_env/tacto && \
    PIP_CONSTRAINT=/opt/gr1-constraints.txt pip install --no-deps -e /repos/calvin/calvin_env && \
    PIP_CONSTRAINT=/opt/gr1-constraints.txt pip install --no-deps -e /repos/calvin/calvin_models

# Minimal ROCm/API source patches on top of the pinned upstream (rule 2.1). Applied after the
# pip layers (patches only touch the git checkouts) so patch edits don't invalidate deps caching.
#  * rocm_port.py  -> GR-1 (device masks, map_location, GPT-2 docstring kwarg drift)
#  * calvin_port.py -> calvin_env camera RGB to uint8 (pybullet 3.2.7 + numpy 2.x returns int64)
COPY patches/ /tmp/gr1-patches/
RUN python /tmp/gr1-patches/rocm_port.py /repos/gr1 && \
    python /tmp/gr1-patches/calvin_port.py /repos/calvin

# Fail early if the CALVIN import surface the closed-loop runner uses is broken.
RUN python -c "from calvin_env.envs.play_table_env import get_env; from calvin_agent.models.calvin_base_model import CalvinBaseModel; from calvin_env.envs.tasks import Tasks; import pybullet, quaternion, hydra; print('OK calvin imports')"

# Headless EGL rendering defaults + CALVIN config root (used by demos/demo_calvin.py).
ENV PYTHONPATH=/repos/gr1 \
    CALVIN_ROOT=/repos/calvin \
    MESA_GL_VERSION_OVERRIDE=3.3 \
    PYOPENGL_PLATFORM=egl

COPY test.py /ryzers/
COPY scripts/ /ryzers/scripts/
COPY demos/ /ryzers/demos/
RUN chmod +x /ryzers/scripts/*.sh /ryzers/demos/*.sh 2>/dev/null || true

CMD ["python", "/ryzers/test.py"]
