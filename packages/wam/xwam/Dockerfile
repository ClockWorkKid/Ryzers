# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# X-WAM (Wan2.2-TI2V-5B unified 4D world-action model) policy/model layer for AMD Strix
# Halo (gfx1151). This is a slim policy image: it installs only the X-WAM repo on the base
# image's ROCm torch (upstream never installs its own torch — the torch/flash-attn lines in
# requirements.txt are commented out — so the base ROCm torch is preserved) plus its runtime
# deps. It ships NO simulator. It composes on:
#   * the plain ROCm base   -> non-sim demos (smoke / latency / open-loop / videogen)
#   * simulation/robotwin   -> closed-loop + interactive RoboTwin (ryzers build robotwin xwam)
#   * simulation/robocasa   -> closed-loop + interactive RoboCasa (new base; ryzers build robocasa xwam)
# Installing under a constraint pinning the base's torch lets the exact same layer compose on
# a plain base and on a sim base without replacing the ROCm torch stack. numpy is left free to
# resolve to X-WAM's required <2 (matches the sim bases' numpy 1.26.x). Weights (Wan2.2 base +
# X-WAM checkpoints), datasets and sim assets are fetched at runtime / supplied by the sim base.

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG XWAM_COMMIT=72cfb86b33fc5060963ef63412f16439fcfa472f

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    PYTHONPATH=/repos/xwam

WORKDIR /ryzers

RUN apt-get update && \
    apt-get install -y --no-install-recommends git cmake build-essential ffmpeg libgl1 && \
    rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/sharinka0715/X-WAM /repos/xwam && \
    git -C /repos/xwam checkout ${XWAM_COMMIT}

# Pin the base's ROCm torch so the X-WAM install (and its transitive deps: diffusers,
# transformers, deepspeed, lightning, ...) never replaces it with a CUDA wheel. X-WAM's
# requirements.txt already comments out the torch* and flash-attn lines, so nothing tries to
# install torch directly; the constraint is belt-and-suspenders against transitive upgrades.
# flash-attn is intentionally NOT installed — modules/attention.py falls back to torch SDPA
# on gfx1151 (flash-attn has no ROCm/gfx1151 wheel).
#
# numpy is pinned to 1.26.4: X-WAM requires numpy<2 (>=1.23.5) and the simulation bases ship
# numpy 1.26.4 for their numpy-1.x-ABI native extensions (e.g. RoboTwin's mplib 0.2.1
# `convert_physx_component` segfaults under numpy 2.x). The plain ROCm base ships numpy 2.x, so
# it is pinned down here; the constraint then protects the pinned numpy + base ROCm torch from
# transitive upgrades. X-WAM's model path is unaffected (supports numpy>=1.23.5).
RUN pip install --no-cache-dir "numpy==1.26.4" && \
    pip freeze | grep -iE "^(torch|torchvision|torchaudio|torchcodec|numpy)==" > /opt/xwam-constraints.txt && \
    cat /opt/xwam-constraints.txt && \
    PIP_CONSTRAINT=/opt/xwam-constraints.txt pip install --no-cache-dir -r /repos/xwam/requirements.txt && \
    PIP_CONSTRAINT=/opt/xwam-constraints.txt pip install --no-cache-dir \
      "hf-transfer>=0.1.8" "huggingface_hub>=0.25" "pyzmq>=25" "matplotlib>=3.8" "opencv-python-headless" && \
    pip install --no-cache-dir "numpy==1.26.4"

# Build-time sign-of-life: ROCm torch preserved + the heavy runtime deps import on it. We do
# NOT import the X-WAM `modules` package here: modules/t5.py touches torch.cuda.current_device()
# at class-definition time and `docker build` has no GPU. The full X-WAM import graph (+ the
# attention SDPA fallback) is validated at run time by test.py, which runs with the iGPU via
# `ryzers run` (flash-attn has no gfx1151 wheel, so SDPA is the path).
RUN python -c "import torch; assert torch.version.hip, 'torch is not a ROCm build: '+torch.__version__; import diffusers, transformers, lightning, einops, omegaconf, zmq, cv2; print('OK xwam deps on torch', torch.__version__, 'hip', torch.version.hip, '| transformers', transformers.__version__, '| diffusers', diffusers.__version__)"

COPY test.py /ryzers/
COPY scripts/ /ryzers/scripts/
COPY demos/ /ryzers/demos/
# Closed-loop RoboTwin plugin (direct in-process X-WAM policy). RoboTwin's script/eval_policy.py
# imports it via a symlink into <ROBOTWIN_ROOT>/policy created by eval_robotwin_single.py; only
# used when this layer is chained on the simulation/robotwin base (ryzers build robotwin xwam).
COPY experiments/ /ryzers/experiments/
RUN chmod +x /ryzers/scripts/*.sh /ryzers/demos/*.sh 2>/dev/null || true

CMD ["python", "/ryzers/test.py"]
