# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Latte (Vchitect/Latte) — latent-diffusion video-generation DiT — for AMD Strix Halo
# (gfx1151). Installs Latte's upstream code on the base image's ROCm torch. Upstream is
# conda/CUDA (environment.yml: pytorch-cuda + nvidia channel); we drop conda + pytorch-cuda,
# keep the base ROCm torch, and translate the `pip:` section into a constrained pip install.
#
# Two generation paths (both near-pure PyTorch, run on the same image):
#  * class-conditional (sample/sample.py + diffusion/ DDPM/DDIM, models/latte.py): ffs, sky,
#    ucf101, taichi. Only needs diffusers AutoencoderKL (version-agnostic).
#  * T2V / T2I (sample/sample_t2x.py + sample/pipeline_latte.py + models/latte_t2v.py, T5 +
#    SD-VAE): t2v, t2i. models/latte_t2v.py imports diffusers-0.24.0-era internal APIs
#    (CaptionProjection, LoRACompatibleConv/Linear, CombinedTimestepSizeEmbeddings) that newer
#    diffusers renamed/removed -> we honor upstream's diffusers==0.24.0 pin (rule 2.1) and pin a
#    compatible huggingface_hub/transformers/tokenizers set. diffusers 0.24.0 needs
#    huggingface_hub.constants.hf_cache_home (present in hub 0.20.x, dropped in newer hub).
#
# ROCm source patches (patches/rocm_port.py), all minimal + idempotent (rule 2.1):
#  * latte.py/latte_img.py `flash` attention: drop the forced enable_math=False sdp_kernel
#    (unsupported on gfx1151) AND add the `.transpose(1, 2)` the upstream flash branch omitted, so
#    SDPA is a numerically-exact (cos-sim 1.0), ~1.8x-faster drop-in for `math` (see docs/OPTIMIZATIONS.md).
#  * metrics eval stack: fix the stale `training.dataset.ImageFolderDataset` class name, register a
#    16-frame `fvd_16f`, and split IS over the actual sample count (small-N NaN fix).
# `math` attention mode is the untouched default. Weights (HF maxin-cn/Latte-1, maxin-cn/Latte,
# SD-VAE) are fetched at runtime into the mounted HF cache (rule 8).

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG LATTE_COMMIT=b27c24a17487e3c33ae4e92364521303d0f14845

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /ryzers

RUN apt-get update && \
    apt-get install -y --no-install-recommends git cmake build-essential ffmpeg && \
    rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/Vchitect/Latte /repos/latte && \
    git -C /repos/latte checkout ${LATTE_COMMIT}

# Minimal ROCm source patches on top of the pinned upstream (rule 2.1). See patches/rocm_port.py.
COPY patches/ /tmp/latte-patches/
RUN python /tmp/latte-patches/rocm_port.py /repos/latte

# Pin the base's torch + torchvision + numpy so nothing pulls a CUDA build or breaks the torch
# ABI. Then install Latte's environment.yml `pip:` section under that constraint, honoring
# upstream's diffusers==0.24.0 with a matching hub/transformers/tokenizers set.
RUN pip freeze | grep -iE "^(torch|torchvision|numpy)==" > /opt/latte-constraints.txt && \
    PIP_CONSTRAINT=/opt/latte-constraints.txt pip install --no-cache-dir \
      "diffusers[torch]==0.24.0" \
      "huggingface_hub==0.20.3" \
      "transformers==4.36.2" \
      "tokenizers==0.15.2" \
      "accelerate==0.25.0" \
      "safetensors" \
      "timm==0.9.16" \
      "einops" "omegaconf" "sentencepiece" "beautifulsoup4" "ftfy" \
      "av" "decord" "imageio" "imageio-ffmpeg" "scikit-image" "pandas" \
      "tensorboard" "opencv-python-headless" "matplotlib"

# ROCm fix (rule 2.1): the base image ships apex (apex 1.10.0+rocm). transformers' T5 replaces
# its native T5LayerNorm with apex.normalization.FusedRMSNorm when apex imports, but that fused
# kernel raises "expected scalar type Float but found Half" on fp16 (the T2V/T2I text encoder
# runs fp16). Latte uses no apex, so remove it -> transformers falls back to its fp16-safe
# native T5LayerNorm.
RUN pip uninstall -y apex || true

# Fail early if the torch stack was clobbered to a non-ROCm build by the deps above.
RUN python -c "import torch; assert torch.version.hip, 'torch is not a ROCm build: '+torch.__version__; print('OK torch', torch.__version__, 'hip', torch.version.hip)"

ENV PYTHONPATH=/repos/latte

COPY test.py /ryzers/
COPY scripts/ /ryzers/scripts/
COPY demos/ /ryzers/demos/
RUN chmod +x /ryzers/scripts/*.sh /ryzers/demos/*.sh 2>/dev/null || true

CMD ["python", "/ryzers/test.py"]
