# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Cosmos3-Nano-Policy-DROID (16B Mixture-of-Transformers World-Action Model) policy/model
# layer for AMD Strix Halo (gfx1151). Slim policy image: installs the NVIDIA cosmos-framework
# policy/inference stack on the base image's ROCm torch (upstream CUDA torch + NVIDIA-only
# attention/kernels are never pulled -- see the install-strategy note below). Ships NO simulator.
#   * plain ROCm base -> smoke / forward / DROID open-loop (simulator-free; the current scope)
#   * closed-loop is DEFERRED: RoboLab = Isaac Sim (proprietary). ROCm-friendly alt = the
#     upstream LIBERO HTTP server on packages/simulation/libero (see docs/SCOPING.md).
#
# ---------------------------------------------------------------------------------------------
# Phase 1 install strategy (validated against the real pyproject @ pin 0fa3ba47):
#   * cosmos-framework's [project].dependencies contain NO torch/numpy/transformer-engine/
#     natten/flash-attn -- those live only in the [dependency-groups] (cu130-train, vllm) and
#     [project.optional-dependencies].train, which pip does NOT read for a plain `pip install`.
#   * So we install the core deps faithfully with `pip install -e .` (no pyproject mutation),
#     under a PIP_CONSTRAINT that pins the base image's ROCm torch/torchvision/numpy so a
#     transitive pull (diffusers/transformers/accelerate depend on torch) can never replace
#     the ROCm build. Attention routes to SDPA/AOTriton on gfx1151.
#   * transformer_engine / natten / flash_attn are referenced only in train/other-modality
#     code paths. If `import cosmos_framework` (smoke) trips a hard import, a runtime fallback
#     shim goes in patches/ (kept out of the build so the image builds and we iterate on smoke).
# ---------------------------------------------------------------------------------------------

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG COSMOS_COMMIT=0fa3ba479a7c1c5be28f81fe084b9ea544d697c9

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    # cosmos-framework quirk: clear host lib bleed-through after env setup.
    LD_LIBRARY_PATH= \
    # Force SDPA/AOTriton attention on gfx1151 (no natten/flash-attn wheels).
    TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1

WORKDIR /ryzers

RUN apt-get update && \
    apt-get install -y --no-install-recommends git git-lfs cmake build-essential ffmpeg curl && \
    rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/NVIDIA/cosmos-framework /repos/cosmos-framework && \
    git -C /repos/cosmos-framework checkout ${COSMOS_COMMIT}

# NOTE (Phase 1): if smoke trips a hard NVIDIA-only import, drop a patch here BEFORE install:
#   - transformer_engine layers -> torch.nn / SDPA equivalents
#   - natten (neighborhood attention, video tower) -> SDPA fallback
#   - flash_attn imports -> guarded (upstream notes flash-attn deserialization is unused here)
# COPY patches/ /tmp/cosmos-patches/
# RUN for p in /tmp/cosmos-patches/*.patch; do git -C /repos/cosmos-framework apply -v "$p"; done

# Pin the base image's torch/torchvision/numpy so the cosmos install (and its transitive
# torch-dependent deps) can never replace the ROCm torch build nor force a numpy downgrade.
# Then install cosmos-framework's [project].dependencies faithfully (no --no-deps, no extras,
# no dependency-groups -> the CUDA/TE/natten/flash-attn pins are never seen by pip).
RUN pip freeze | grep -iE "^(torch|torchvision|torchaudio|torchcodec|numpy|pytorch-triton-rocm)==" > /opt/cosmos-constraints.txt && \
    echo "=== ROCm constraints held ===" && cat /opt/cosmos-constraints.txt && \
    PIP_CONSTRAINT=/opt/cosmos-constraints.txt pip install --no-cache-dir -e /repos/cosmos-framework && \
    PIP_CONSTRAINT=/opt/cosmos-constraints.txt pip install --no-cache-dir "hf-transfer>=0.1.8" safetensors

# Inference/eval runtime deps for the DROID action-policy path. Discovered empirically by
# walking the pinned cosmos-framework import chain (inference.inference / omni_mot_model /
# action datasets / easy_io storage backends) on ROCm; none of transformer_engine / natten /
# flash-attn are hard imports on this path. Installed under the same ROCm torch/numpy
# constraint so the base torch is never disturbed. lerobot is the mli0603 fork the pyproject pins.
RUN PIP_CONSTRAINT=/opt/cosmos-constraints.txt pip install --no-cache-dir \
      qwen-vl-utils iopath pyarrow mediapy einx webdataset \
      multi-storage-client boto3 datasets google-cloud-storage \
      "lerobot @ git+https://github.com/mli0603/lerobot.git"

# Build-time guard: ROCm torch present (no GPU needed at build).
RUN python -c "import torch; assert torch.version.hip, 'torch is not a ROCm build: '+torch.__version__; print('OK torch', torch.__version__, 'hip', torch.version.hip)"

COPY test.py /ryzers/
COPY scripts/ /ryzers/scripts/
COPY demos/ /ryzers/demos/
RUN chmod +x /ryzers/scripts/*.sh /ryzers/demos/*.sh || true

CMD ["python", "/ryzers/test.py"]
