# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# AVDC (flow-diffusion) as the iTHOR ObjectNav MODEL layer for AMD Strix Halo (gfx1151), built ON
# TOP of the model-agnostic AI2-THOR simulator base:
#
#     ryzers build ithor avdc-ithor --name avdc-ithor
#
# The sim base (packages/simulation/ithor) already provides ai2thor + the Unity CloudRendering
# (Vulkan) runtime + the sim_ithor harness (ThorEnv / Policy seam / rollout). Here we add ONLY the
# AVDC model half: the flowdiffusion 3D-UNet video-diffusion policy (get_video_model_thor,
# milestone-16), the UniMatch (GMFlow) dense-flow model, CLIP ViT-B/32, and the
# video->flow->transforms->nav-actions pipeline (experiment/myutils + thor_exp). A thin
# sim_ithor.Policy adapter (avdc_ithor_policy) wires them behind the seam. Weights fetched at
# runtime (rule 8). Same model stack as the Meta-World AVDC image, so avdc_optim (fp16 +
# torch.compile) reuses directly.

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG AVDC_COMMIT=e9b5b0098e59da0e1fe24a18ed9bec5b825e2ca2

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /ryzers

# Sanity: this MUST be built on the simulation/ithor base (it supplies ai2thor + sim_ithor).
RUN python -c "import sim_ithor, ai2thor" || { \
      echo "ERROR: build FROM the simulation/ithor base: ryzers build ithor avdc-ithor"; \
      exit 1; }

RUN apt-get update && \
    apt-get install -y --no-install-recommends git cmake build-essential ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# AVDC codebase (model + experiment/ closed-loop utils: myutils flow->actions, thor_exp) at the
# same pin as the Meta-World AVDC package.
RUN git clone https://github.com/flow-diffusion/AVDC_experiments /repos/avdc && \
    git -C /repos/avdc checkout ${AVDC_COMMIT}

# Pin the base's torch/torchvision/numpy so nothing pulls a CUDA build or breaks the ABI, then
# install AVDC's model-path deps under that constraint (matches the Meta-World AVDC image). numpy
# stays at the base's 1.26.4. flowdiffusion (with vendored guided_diffusion) is editable, no-deps.
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

# UniMatch (GMFlow) dense-flow weights for the video->flow->nav pipeline, fetched from the upstream
# unimatch S3 at build (rule 8). get_flow_model() reads pretrained/ relative to experiment/.
RUN mkdir -p /repos/avdc/experiment/pretrained && \
    wget -q -O /repos/avdc/experiment/pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth \
      https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth

# Minimal ROCm source patch (NVML no-op) on the pinned upstream (rule 2.1). Late layer so it does
# not invalidate the pip cache.
COPY patches/ /tmp/avdc-patches/
RUN python /tmp/avdc-patches/rocm_port.py /repos/avdc

# flowdiffusion (model) + experiment/ (myutils flow->actions, thor_exp) on PYTHONPATH; the AVDC
# adapter goes on PYTHONPATH so POLICY_FACTORY=avdc_ithor_policy:build_policy resolves against the
# sim base harness. avdc_optim shim lives in /ryzers.
ENV PYTHONPATH=/repos/avdc/flowdiffusion:/repos/avdc/experiment:/opt/avdc-adapters:/ryzers:${PYTHONPATH}

COPY avdc_optim.py /ryzers/
COPY adapters/ /opt/avdc-adapters/
COPY scripts/ /ryzers/scripts/
COPY demos/ /ryzers/demos/
RUN chmod +x /ryzers/scripts/*.sh /ryzers/demos/*.sh 2>/dev/null || true

RUN python -c "import sim_ithor, torch; from flowdiffusion.inference_utils import get_video_model_thor, pred_video_thor; \
from myutils import get_flow_model, pred_flow_frame, get_transforms_nav, transforms2actions; \
import avdc_ithor_policy; print('OK avdc-ithor model layer on torch', torch.__version__, 'hip', torch.version.hip)"

CMD ["python", "/ryzers/demos/demo_ithor.py", "--help"]
