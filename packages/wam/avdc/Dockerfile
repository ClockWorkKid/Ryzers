# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# AVDC (flow-diffusion) - "Learning to Act from Actionless Videos through Dense Correspondences" -
# a text-conditioned 3D-UNet video-diffusion policy, for AMD Strix Halo (gfx1151, ROCm). This is a
# SINGLE consolidated AVDC image covering all three demos in one build:
#   * open-loop video prediction (sim-agnostic),
#   * closed-loop Meta-World  (vendored MuJoCo, headless EGL),
#   * closed-loop iTHOR ObjectNav (ai2thor CloudRendering / Vulkan, via the sim base).
#
# It is built ON TOP of the model-agnostic AI2-THOR simulator base so the iTHOR half reuses that
# base's ai2thor + Vulkan runtime + sim_ithor harness (rather than duplicating it):
#
#     ryzers build ithor avdc --name avdc
#
# On top of that base we add (a) the AVDC model stack (flowdiffusion 3D-UNet + CLIP ViT-B/32 +
# GoalGaussianDiffusion DDIM sampler + UniMatch GMFlow), and (b) the vendored Meta-World MuJoCo
# simulator for the closed-loop MW benchmark. Weights (CLIP + AVDC snapshots) are fetched at
# runtime (rule 8); UniMatch + MuJoCo runtime are fetched from upstream at build (rule 8).

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG AVDC_COMMIT=e9b5b0098e59da0e1fe24a18ed9bec5b825e2ca2

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /ryzers

# Sanity: this MUST be built on the simulation/ithor base (it supplies ai2thor + sim_ithor).
RUN python -c "import sim_ithor, ai2thor" || { \
      echo "ERROR: build FROM the simulation/ithor base: ryzers build ithor avdc"; \
      exit 1; }

# Build tools + the Meta-World / MuJoCo GL stack (libs resolve via ldconfig at runtime).
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      git cmake build-essential ffmpeg \
      libgl1-mesa-dev libglew-dev libosmesa6-dev libglfw3 patchelf wget && \
    rm -rf /var/lib/apt/lists/*

# AVDC codebase (model + experiment/: myutils flow->actions, thor_exp, vendored Meta-World).
RUN git clone https://github.com/flow-diffusion/AVDC_experiments /repos/avdc && \
    git -C /repos/avdc checkout ${AVDC_COMMIT}

# Pin the base's torch/torchvision/numpy so nothing pulls a CUDA build or breaks the ABI, then
# install AVDC's model-path deps under that constraint. numpy stays at the base's 1.26.4.
# flowdiffusion (with vendored guided_diffusion) is installed editable, no-deps.
RUN pip freeze | grep -iE "^(torch|torchvision|numpy)==" > /opt/avdc-constraints.txt && \
    cat /opt/avdc-constraints.txt && \
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
# Meta-World closed-loop simulator layer - vendored MuJoCo (mujoco-py / MuJoCo 2.1) + gym. On
# gfx1151 mujoco-py renders headless via EGL on the iGPU (physics on CPU, offscreen GL on GPU).
# Only add mujoco210's lib dir; ROCm + system GL libs resolve via ldconfig. Do NOT prepend
# /usr/lib/x86_64-linux-gnu here - it shadows ROCm's libhsa-runtime64 and breaks the torch import.
# ---------------------------------------------------------------------------------------------
ENV MUJOCO_GL=egl \
    MUJOCO_PY_MUJOCO_PATH=/root/.mujoco/mujoco210 \
    LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin

# MuJoCo 2.1 runtime (fetched from the upstream DeepMind release at build - rule 8, not re-hosted).
RUN mkdir -p /root/.mujoco && cd /root/.mujoco && \
    wget -q https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz && \
    tar -xzf mujoco210-linux-x86_64.tar.gz && rm mujoco210-linux-x86_64.tar.gz

# The Meta-World / mujoco-py / gym 0.25 stack predates numpy 2; cython<3 is needed to cythonize
# mujoco-py on py3.12; gym 0.25.2 keeps the classic API Meta-World expects. numpy stays 1.26.4
# (the base pin); pinning again here is a no-op guard.
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

# Minimal ROCm source patch (NVML no-op) on the pinned upstream (rule 2.1). Late layer so it does
# not invalidate the pip cache.
COPY patches/ /tmp/avdc-patches/
RUN python /tmp/avdc-patches/rocm_port.py /repos/avdc

# flowdiffusion (model) + experiment/ (myutils flow->actions, thor_exp, metaworld_exp) on
# PYTHONPATH; the iTHOR adapter goes on PYTHONPATH so POLICY_FACTORY=avdc_ithor_policy:build_policy
# resolves against the sim base harness. avdc_optim shim lives in /ryzers.
ENV PYTHONPATH=/repos/avdc/flowdiffusion:/repos/avdc/experiment:/opt/avdc-adapters:/ryzers:${PYTHONPATH}

COPY avdc_optim.py /ryzers/
COPY adapters/ /opt/avdc-adapters/
COPY scripts/ /ryzers/scripts/
COPY demos/ /ryzers/demos/
# AVDC model forward smoke (weight-free); kept alongside the base's Vulkan sim gate (/ryzers/test.py).
COPY test.py /ryzers/model_smoke.py
RUN chmod +x /ryzers/scripts/*.sh /ryzers/demos/*.sh 2>/dev/null || true

# Consolidated smoke: both simulators + the AVDC model stack import in one image.
RUN python -c "import sim_ithor, mujoco_py, torch; \
from flowdiffusion.inference_utils import get_video_model, get_video_model_thor; \
from myutils import get_flow_model, pred_flow_frame, get_transforms_nav, transforms2actions; \
import avdc_ithor_policy; \
print('OK avdc (MW+iTHOR) on torch', torch.__version__, 'hip', torch.version.hip)"

CMD ["python", "/ryzers/model_smoke.py"]
