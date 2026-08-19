# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# FlowWAM (Wan2.2-TI2V-5B dual-stream world-action model) model layer for AMD Strix Halo
# (gfx1151). FlowWAM uses dense optical flow as a unified action representation: a dual-stream
# Wan DiT jointly generates a future RGB video + flow field, with RAFT + a reversible flow codec
# and a SAPIEN robot-only renderer (RoboTwin2.0 embodiments). Upstream is a trimmed DiffSynth
# (installed as the `diffsynth` package) -- the same Wan-video framework FastWAM builds on, so
# this reuses FastWAM's ROCm playbook: strip base-owned CUDA/torch/numpy/sapien pins, hold them
# via PIP_CONSTRAINT, install upstream on the base's ROCm torch. No flash-attn/apex/cublas are
# installed: DiffSynth's flash_attention() falls back to torch SDPA on ROCm, and apex is only
# needed by the (deferred) SeedVR refiner.
#
# FlowWAM requires SAPIEN robot rendering + RoboTwin embodiments, so it chains on the
# simulation/robotwin sim base (Vulkan SAPIEN headless on gfx1151):
#   ryzers build robotwin flowwam --name flowwam-robotwin
# Weights (Wan2.2-TI2V-5B, UMT5-XXL enc, FlowWAM stage-1/robotwin, embodiments) are fetched at
# run time by scripts/, never re-hosted. The SeedVR2 refiner is intentionally out of this image.

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG FLOWWAM_COMMIT=f06fa46042e97738c6619c868f1097be6749d48d
ARG FLOWWAM_FULL_COMMIT=68abaa2

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    FLOWWAM_REPO=/repos/flowwam \
    FLOWWAM_FULL_REPO=/repos/flowwam-full

WORKDIR /ryzers

RUN apt-get update && \
    apt-get install -y --no-install-recommends git cmake build-essential ffmpeg unzip && \
    rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/YixiangChen515/FlowWAM_WorldArena /repos/flowwam && \
    git -C /repos/flowwam checkout ${FLOWWAM_COMMIT}

# Pin the base's torch + numpy + sapien so the FlowWAM install never replaces them; strip those
# (and any defensive CUDA-only) lines from requirements.txt so the robotwin base's ROCm torch +
# numpy 1.26 + Vulkan SAPIEN stay intact while the rest of diffsynth's deps install.
COPY scripts/strip_cuda_torch.py /ryzers/scripts/strip_cuda_torch.py
RUN python /ryzers/scripts/strip_cuda_torch.py /repos/flowwam/requirements.txt && \
    pip freeze | grep -iE "^(torch|torchvision|torchcodec|numpy|sapien)==" > /opt/flowwam-constraints.txt && \
    cat /opt/flowwam-constraints.txt && \
    PIP_CONSTRAINT=/opt/flowwam-constraints.txt pip install --no-cache-dir -r /repos/flowwam/requirements.txt && \
    PIP_CONSTRAINT=/opt/flowwam-constraints.txt pip install --no-cache-dir -e /repos/flowwam && \
    PIP_CONSTRAINT=/opt/flowwam-constraints.txt pip install --no-cache-dir \
      "hf-transfer>=0.1.8" "huggingface-hub>=0.34" "imageio-ffmpeg>=0.5"

# Closed-loop layer: the genuine flow-ACTION loop (IDM action expert -> RoboTwin env.step over a
# ws server + robotwin_policy) lives in the full FlowWAM repo. Clone it at a pin and add only its
# thin transport deps; its diffsynth is used by the server via runtime PYTHONPATH (see
# demos/demo_closedloop_robotwin.sh), NOT pip-installed, so the validated open-loop diffsynth above
# is left untouched.
RUN git clone https://github.com/YixiangChen515/FlowWAM.git /repos/flowwam-full && \
    git -C /repos/flowwam-full checkout ${FLOWWAM_FULL_COMMIT} 2>/dev/null || true
RUN PIP_CONSTRAINT=/opt/flowwam-constraints.txt pip install --no-cache-dir \
      "websockets>=12.0" "msgpack>=1.0" "msgpack-numpy>=0.4.8" \
      "sentencepiece>=0.2" "tiktoken>=0.7"

# Import + env sign-of-life at build time (fails the build early if the layer is broken).
RUN python -c "import torch; assert torch.version.hip, 'torch is not a ROCm build: '+torch.__version__; \
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig; \
from diffsynth.models.wan_video_dit_dual_stream import init_flow_stream; \
from diffsynth.pipelines.wan_video_dual_stream import model_fn_wan_video_dual_stream; \
import sapien, cv2, h5py, accelerate; \
print('OK flowwam/diffsynth on torch', torch.__version__, 'hip', torch.version.hip)"

COPY test.py /ryzers/
COPY scripts/ /ryzers/scripts/
COPY demos/ /ryzers/demos/
RUN chmod +x /ryzers/scripts/*.sh /ryzers/demos/*.sh 2>/dev/null || true

CMD ["python", "/ryzers/test.py"]
