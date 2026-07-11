# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# VLA-JEPA (ginwind/VLA-JEPA, ECCV 2026) on AMD Strix Halo (gfx1151) Ryzer package.
# Layers the VLA-JEPA inference stack on the default Ryzer ROCm base
# (rocm/pytorch, torch 2.10.0+rocm7.2.2). VLA-JEPA = Qwen3-VL VLM + GR00T
# flow-matching action head, regularized at train time by a V-JEPA2 latent world
# model. Built on the starVLA codebase.
#
# This layer covers capability 1 (environment + full-model smoke). Open-loop
# replay and closed-loop LIBERO/SimplerEnv are added in later commits.
#
# We keep the base image's ROCm torch/torchvision (upstream's CUDA-only pins are
# NOT installed); a build-time guard fails if torch is swapped off ROCm. The one
# source patch we apply is a ROCm bug fix: Qwen3-VL is hardcoded to
# flash_attention_2 (a CUDA-only wheel) upstream, so we flip it to SDPA.

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# Pin the upstream VLA-JEPA source this package builds against.
ARG VLAJEPA_COMMIT=ec8c70f6e155e2377bbd4d787004c14179c00c7c

WORKDIR /ryzers

ENV HF_HUB_ENABLE_HF_TRANSFER=0 \
    HF_HUB_DISABLE_XET=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false

# VLA-JEPA / starVLA runtime deps EXCEPT torch/torchvision (CUDA wheels would
# clobber the ROCm torch already in the base image), flash-attn (CUDA-only; we
# patch to SDPA), deepspeed (training only), and pytorch3d/decord (only used by
# the LeRobot dataloader transforms / video decode paths, not the inference
# smoke; added later when the open-loop path needs them).
RUN pip install --no-cache-dir \
    "transformers==4.57.0" \
    "accelerate==1.5.2" \
    "huggingface-hub[cli]>=0.36,<1.0" \
    "hf-transfer>=0.1.8" \
    "safetensors>=0.4" \
    "sentencepiece>=0.2" \
    "tiktoken" \
    "einops" \
    "transformers_stream_generator==0.0.4" \
    "qwen-vl-utils" \
    "omegaconf" \
    "pydantic==2.10.6" \
    "numpydantic==1.6.9" \
    "numpy==1.26.4" \
    "scipy" \
    "pillow" \
    "matplotlib" \
    "h5py" \
    "rich" \
    "diffusers" \
    "timm" \
    "albumentations==1.4.18" \
    "av==12.3.0" \
    "pyarrow==14.0.1" \
    "fastparquet==2024.11.0" \
    "websocket-client==1.8.0" \
    "websockets" \
    "msgpack" \
    "imageio>=2.34" \
    "imageio-ffmpeg>=0.5"

# Bake the upstream VLA-JEPA source at the pinned commit and install the starVLA
# package itself WITHOUT deps (its pyproject declares no deps; deps are the pinned
# set above), so pip cannot pull a CUDA torch.
RUN export DEBIAN_FRONTEND=noninteractive && apt-get update && \
    apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/* && \
    rm -rf /repos/VLA-JEPA && \
    git clone https://github.com/ginwind/VLA-JEPA /repos/VLA-JEPA && \
    git -C /repos/VLA-JEPA checkout ${VLAJEPA_COMMIT} && \
    pip install --no-cache-dir --no-deps -e /repos/VLA-JEPA

# ROCm bug patch (rule 2.1): upstream hardcodes attn_implementation="flash_attention_2"
# in the Qwen VLM interfaces, which needs the CUDA-only flash-attn wheel. Flip to
# "sdpa" (PyTorch scaled_dot_product_attention; aotriton-backed on gfx1151).
COPY scripts/apply_rocm_patches.py /ryzers/apply_rocm_patches.py
RUN python /ryzers/apply_rocm_patches.py

# Build-time guard: fail the build if anything swapped torch off ROCm.
RUN python -c "import torch; assert torch.version.hip, 'torch is not a ROCm build: '+torch.__version__; print('OK torch', torch.__version__, 'hip', torch.version.hip)"

# Demo entrypoints + smoke + open-loop replay + latent-imagination + HF helpers.
COPY test.py /ryzers/
COPY scripts/model_smoke.py scripts/openloop_replay.py scripts/imagine_latent.py /ryzers/
COPY scripts/closedloop_libero.py scripts/closedloop_liberoplus.py scripts/closedloop_simplerenv.py /ryzers/
COPY demos/demo_smoke.sh demos/demo_openloop.sh demos/demo_imagine.sh /ryzers/
COPY scripts/_hf_common.sh scripts/download.sh scripts/download_smoke.sh scripts/download_openloop.sh /ryzers/

# LIBERO / LIBERO-Plus / SimplerEnv closed-loop / interactive demos + the sim_*.Policy adapters.
# These only do anything when this layer is chained on the matching simulation base
# (ryzers build libero vlajepa / libero-plus vlajepa / simulation/simplerenv vlajepa): each
# adapter imports its sim harness (sim_libero / sim_liberoplus / sim_simplerenv), and the demos
# add it to PYTHONPATH and set POLICY_FACTORY. On a non-matching base they are inert files.
COPY adapters/ /opt/vlajepa-adapters/
COPY demos/demo_closedloop_libero.sh demos/demo_interactive_libero.sh demos/demo_interactive_libero_rt.sh /ryzers/demos/
COPY demos/demo_closedloop_liberoplus.sh demos/demo_interactive_liberoplus.sh demos/demo_interactive_liberoplus_rt.sh /ryzers/demos/
COPY demos/demo_closedloop_simplerenv.sh demos/demo_interactive_simplerenv.sh demos/demo_interactive_simplerenv_rt.sh /ryzers/demos/
RUN chmod +x /ryzers/*.sh /ryzers/demos/*.sh

# Default `ryzers run` (no CMD override) = lightweight env sign-of-life (no weights).
CMD ["python", "/ryzers/test.py"]
