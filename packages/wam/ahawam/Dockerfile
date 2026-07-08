# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# AHA-WAM (async Wan2.2-TI2V-5B world-action model) policy/model layer for AMD Strix Halo
# (gfx1151). This is a slim policy image: it installs only the AHA-WAM model on the base
# image's ROCm torch (upstream CUDA torch pins are stripped so the base ROCm torch is
# preserved). It ships NO simulator. It composes on:
#   * the plain ROCm base  -> non-sim demos (smoke / latency / open-loop / async real-time)
#   * simulation/robotwin  -> closed-loop RoboTwin 2.0 (primary target)
# Installing under a constraint pinning the base's torch+numpy lets the exact same layer
# compose on a plain base (numpy 2.x) or on a sim base (numpy 1.26.4) without conflict.
# Weights, datasets and sim assets are fetched at runtime / supplied by the sim base.
# AHA-WAM is the direct successor of FastWAM; this package mirrors packages/wam/fastwam.

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG AHAWAM_COMMIT=69076f6206518554477be5639deb016eee03a9d6

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /ryzers

RUN apt-get update && \
    apt-get install -y --no-install-recommends git cmake build-essential ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Strix Halo efficiency patch on top of the pinned upstream commit: batches the ~90 tiny
# per-layer KV-editor GEMMs into a handful of stacked GEMMs (mot.py) and caches the static
# instruction text embedding across video prefills (deploy servers + the RoboTwin plugin) so
# umt5 only runs when the instruction changes. Both are lossless (bf16-equivalent) and default
# on (AHAWAM_KV_EDITOR_FAST=1 / AHAWAM_CACHE_TEXT_CONTEXT=1). Generated against ${AHAWAM_COMMIT}.
COPY patches/ /ryzers/patches/
RUN git clone https://github.com/serene-sivy/AHA-WAM /repos/ahawam && \
    git -C /repos/ahawam checkout ${AHAWAM_COMMIT} && \
    git -C /repos/ahawam apply --whitespace=nowarn /ryzers/patches/ahawam_opt.patch && \
    echo "applied ahawam_opt.patch @ ${AHAWAM_COMMIT:0:8}"

# ROCm port: strip the CUDA torch/torchvision/torchcodec pins (keep base ROCm torch) and
# relax upstream's requires-python <3.11 so it installs on the base image's Python. Pin only
# the base's torch stack for the AHA-WAM install so it can never pull CUDA torch; unlike
# FastWAM, AHA-WAM hard-pins numpy==1.26.4, so we let it install that (it matches the
# simulation/robotwin base's numpy exactly), then re-lock torch+numpy for the extras install.
# deepspeed is training-only and dropped for the eval/inference image (its build pulls a CUDA
# toolchain); everything else keeps the exact upstream pins.
COPY scripts/strip_cuda_torch.py /ryzers/scripts/strip_cuda_torch.py
RUN python /ryzers/scripts/strip_cuda_torch.py /repos/ahawam/pyproject.toml && \
    sed -i '/^\s*"deepspeed[=<>!~]/d' /repos/ahawam/pyproject.toml && \
    pip freeze | grep -iE "^(torch|torchvision|torchcodec)==" > /opt/ahawam-constraints.txt && \
    PIP_CONSTRAINT=/opt/ahawam-constraints.txt pip install --no-cache-dir -e /repos/ahawam && \
    pip freeze | grep -iE "^(torch|torchvision|torchcodec|numpy)==" > /opt/ahawam-constraints.txt && \
    PIP_CONSTRAINT=/opt/ahawam-constraints.txt pip install --no-cache-dir \
      "hf-transfer>=0.1.8" "sentencepiece>=0.2" "matplotlib>=3.8" "opencv-python-headless"

# Sim Policy adapter (sim_robotwin.Policy wrapping the AHA-WAM RoboTwin deploy plugin). Only
# imported when driving the simulation/robotwin base via
# POLICY_FACTORY=ahawam_robotwin_policy:build_policy; harmless (unused) on the plain base. On
# its own dir so the interactive demos add it to PYTHONPATH. The RoboTwin closed-loop instead
# uses the repo's own experiments/robotwin/ahawam_policy plugin via RoboTwin's eval_policy.py.
COPY adapters/ /opt/ahawam-adapters/

RUN python -c "import torch; assert torch.version.hip, 'torch is not a ROCm build: '+torch.__version__; import ahawam; from ahawam.runtime import create_ahawam; print('OK ahawam on torch', torch.__version__, 'hip', torch.version.hip)"

COPY test.py /ryzers/
COPY scripts/ /ryzers/scripts/
COPY demos/ /ryzers/demos/
RUN chmod +x /ryzers/scripts/*.sh /ryzers/demos/*.sh

CMD ["python", "/ryzers/test.py"]
