# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# DreamZero (NVIDIA GEAR, Apache-2.0) 14B joint world-action model on AMD Strix
# Halo (gfx1151), packaged as a composable Ryzer. This is the "WAM-direct" port:
# correctness-first, upstream kept pristine, every AMD bridge lives in overlay/.
#
# Design (see README.md + RUNTIME_OPTIMIZATION.md):
#   * Upstream dreamzero is git-cloned at a PINNED commit into /opt/dreamzero and
#     never edited. It is FIRST on PYTHONPATH so `groot.*` imports win.
#   * overlay/python/flash_attn is an SDPA-backed shim satisfying the flash_attn
#     import surface without CUDA-only kernels. With aotriton enabled this routes
#     attention through torch SDPA on ROCm -- no flash-attn source build needed.
#   * All correctness/memory fixes are runtime monkey-patches applied by the
#     overlay (overlay/tests/_stage_b_patches.py, validate_stage3_path_b.py);
#     the upstream tree is byte-for-byte untouched.
#
# Validated on the ROCm base (torch 2.10.0) on gfx1151 under both rocm7.2.2 and
# rocm7.14. The only version-sensitive knob is the allocator: expandable_segments
# must be False on 7.14 (True silently caps the GTT pool at ~19.5 GiB there).
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-lc"]
ENV DEBIAN_FRONTEND=noninteractive

# Reset any entrypoint the base bakes in (it may auto-run an unrelated test).
ENTRYPOINT []
WORKDIR /ryzers

RUN apt-get update \
 && apt-get install -y --no-install-recommends git ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# Overlay wins on PYTHONPATH; every NVIDIA-paper compile optimization is off.
ENV PYTHONPATH=/wam-direct/overlay/python:/opt/dreamzero \
    PYTORCH_ROCM_ARCH=gfx1151 \
    PYTORCH_HIP_ALLOC_CONF=expandable_segments:False \
    TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
    TORCH_BLAS_PREFER_HIPBLASLT=1 \
    ATTENTION_BACKEND=torch \
    FLASH_ATTN_BACKEND=sdpa \
    ENABLE_FLASH_SDPA=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    MIOPEN_USER_DB_PATH=/tmp/miopen \
    MIOPEN_CUSTOM_CACHE_DIR=/tmp/miopen \
    DREAMZERO_COMPILE_TEXT_ENCODER=0 \
    DREAMZERO_COMPILE_IMAGE_ENCODER=0 \
    DREAMZERO_COMPILE_VAE=0 \
    DREAMZERO_COMPILE_DIT=0 \
    TORCHDYNAMO_DISABLE=1 \
    TORCH_COMPILE_DISABLE=1 \
    PYTHONUNBUFFERED=1

# Build-time sanity: torch is the ROCm wheel (guards against a CUDA base swap).
RUN python -c "import torch, sys; v=torch.__version__; hip=torch.version.hip; ok='+rocm' in v and hip is not None; print('torch', v, 'hip', hip, 'ok', ok); sys.exit(0 if ok else 2)"

# Pristine upstream clone at a pinned commit (docs/UPSTREAM_PIN.commit.txt). Kept
# at /opt/dreamzero and NEVER edited; the overlay bridges everything AMD-specific.
ARG DREAMZERO_REPO=https://github.com/dreamzero0/dreamzero.git
ARG DREAMZERO_REF=ab790c198fbce33503358efbbd4187ce9a89adf3
RUN git clone "${DREAMZERO_REPO}" /opt/dreamzero \
 && git -C /opt/dreamzero checkout "${DREAMZERO_REF}" \
 && git -C /opt/dreamzero rev-parse HEAD > /opt/dreamzero.commit \
 && echo "pinned dreamzero @ $(cat /opt/dreamzero.commit)"

# DreamZero inference runtime deps the slim ROCm base does not ship. CUDA-only
# (nvidia-modelopt, tensorrt, flash-attn) and heavy sim/train deps are skipped:
# the overlay SDPA shim replaces flash-attn and the video/open-loop demos are the
# source of truth. The DROID loader reads parquet+mp4 directly (pyarrow + av), so
# `lerobot` is intentionally NOT required.
RUN pip install --no-cache-dir \
        "tianshou==0.5.1" \
        "diffusers==0.30.2" \
        "transformers==4.51.3" \
        "peft==0.5.0" \
        "huggingface-hub<1.0,>=0.30.0" \
        "hf-transfer>=0.1.8" \
        "av==15.0.0" \
        "pyarrow>=17" \
        "albumentations==1.4.18" \
        "hydra-core" "omegaconf" "dm-tree" "draccus" "loguru" "lark" \
        "ftfy" "tiktoken" "sentencepiece" "einops" \
        "imageio>=2.34" "imageio-ffmpeg>=0.5" "matplotlib>=3.8" \
        "opencv-python-headless" "pillow" "psutil" \
 && python -c "import transformers, diffusers, tianshou, av, pyarrow; print('deps OK', transformers.__version__, diffusers.__version__)"

# Bake the AMD overlay (SDPA shim + minimal patch/render suite) so the package is
# self-contained: `ryzers build dreamzero && ryzers run` works with no host mounts.
# During development we instead bind-mount ./overlay read-only for fast iteration.
COPY overlay /wam-direct/overlay

# Ryzers contract: default action validates the install; demos/scripts drive the
# real-weight open-loop + predicted-video (looping-fix) evaluations.
COPY test.py /ryzers/test.py
COPY scripts/ /ryzers/scripts/
COPY demos/ /ryzers/demos/
RUN chmod +x /ryzers/scripts/*.sh /ryzers/demos/*.sh

CMD ["python", "/ryzers/test.py"]
