# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# AMD-AGI/Micro-World — Wan2.1 / VideoX-Fun-derived, action-controlled interactive world model
# (T2W 1.3B + I2W 14B, Minecraft-trained) — for AMD Strix Halo (gfx1151). Installs Micro-World's
# upstream code on the base image's ROCm torch. Same Wan2.1 lineage as VERA, so the benchmark
# optimization recipe transfers directly.
#
# Deps: upstream's requirements.txt/dockerfile pin `diffusers==0.34.0`, `transformers>=4.46.2`,
# `torch>=2.1.2` (no hard CUDA pin, unlike VERA). We keep the base ROCm torch (2.10.0/rocm7.2.2)
# and install a curated INFERENCE subset of the deps under a torch/torchvision PIP_CONSTRAINT so
# nothing swaps in a CUDA wheel or breaks the torch ABI. Training-only heavy deps (gradio, comet_ml,
# tensorboard, datasets, albumentations, deepspeed) are intentionally omitted — inference is scope.
#
# ROCm source patch (patches/rocm_port.py, minimal + idempotent, rule 2.1): the CLIP image encoder's
# AttentionPool calls flash_attention() directly (asserts a flash-attn wheel that gfx1151 lacks); we
# add a torch-SDPA fallback there. The DiT's own attention() dispatcher already SDPA-falls-back, so
# no other change is needed. flash-attn is intentionally NOT installed. Weights (Wan-AI base +
# amd/Micro-World-{T2W,I2W}) are fetched at runtime into the mounted models/HF cache (rule 8).

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG MICRO_WORLD_COMMIT=e63369c88e036756285b089ecc67841c426a7b1b

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /ryzers

RUN apt-get update && \
    apt-get install -y --no-install-recommends git cmake build-essential ffmpeg libheif-dev && \
    rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/AMD-AGI/Micro-World /repos/micro-world && \
    git -C /repos/micro-world checkout ${MICRO_WORLD_COMMIT}

# Minimal ROCm source patch on top of the pinned upstream (rule 2.1). See patches/rocm_port.py.
COPY patches/ /tmp/mw-patches/
RUN python /tmp/mw-patches/rocm_port.py /repos/micro-world

# Pin the base's torch/torchvision/numpy so nothing below pulls a CUDA build or breaks the torch
# ABI, then install the curated inference deps under that constraint (upstream diffusers==0.34.0,
# transformers>=4.46.2). transformers is capped <5: diffusers 0.34.0 imports FLAX_WEIGHTS_NAME,
# which transformers 5.x removed (upstream pinned nothing, but 4.x was current at their build time).
# opencv-python-headless (not opencv-python) keeps this a headless image.
RUN pip freeze | grep -iE "^(torch|torchvision|numpy)==" > /opt/mw-constraints.txt && \
    PIP_CONSTRAINT=/opt/mw-constraints.txt pip install --no-cache-dir \
      "diffusers==0.34.0" \
      "transformers>=4.46.2,<5" \
      "accelerate>=0.25.0" \
      "safetensors" \
      "einops" \
      "omegaconf" \
      "sentencepiece" \
      "ftfy" \
      "Pillow" \
      "scikit-image" \
      "scipy" \
      "imageio" "imageio-ffmpeg" "av" "decord" \
      "torchdiffeq" "torchsde" \
      "timm" "tomesd" \
      "func_timeout" "beautifulsoup4" "pandas" \
      "opencv-python-headless"

# ROCm fix mirrored from the Latte port (rule 2.1): the base image ships apex; transformers can
# swap in apex.normalization.FusedRMSNorm which mishandles half precision. Micro-World uses no apex,
# so remove it to keep transformers on its native, precision-safe norms.
RUN pip uninstall -y apex || true

# Fail early if the torch stack was clobbered to a non-ROCm build by the deps above.
RUN python -c "import torch; assert torch.version.hip, 'torch is not a ROCm build: '+torch.__version__; print('OK torch', torch.__version__, 'hip', torch.version.hip)"

ENV PYTHONPATH=/repos/micro-world

COPY test.py /ryzers/
COPY scripts/ /ryzers/scripts/
COPY demos/ /ryzers/demos/
RUN chmod +x /ryzers/scripts/*.sh /ryzers/demos/*.sh 2>/dev/null || true

CMD ["python", "/ryzers/test.py"]
