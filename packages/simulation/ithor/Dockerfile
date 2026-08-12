# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Model-agnostic AI2-THOR (iTHOR) simulator base image for AMD Strix Halo (gfx1151, ROCm 7.14).
# Ships ONLY the simulator (ai2thor + its Unity CloudRendering build + headless Vulkan runtime) and
# a small model-agnostic harness (sim_ithor): a ThorEnv ObjectNav wrapper, a Policy seam, and a
# generic rollout / sanity runner. No policy or model weights: layer a video/VLA model (AVDC, ...)
# on top with `ryzers build ithor <model>` and select a policy at runtime via
# POLICY_FACTORY (see README).
#
# The crux on gfx1151 is headless Unity rendering: ai2thor renders offscreen through Vulkan
# (platform=CloudRendering, no X server). On AMD that means Mesa RADV (libvulkan1 loader +
# mesa-vulkan-drivers ICD). gfx1151 (RDNA3.5) needs a recent Mesa for RADV, so on Ubuntu 22.04 we
# pull the kisak-mesa PPA. The ai2thor Unity build (~540 MB) is fetched from the allenai public
# bucket at build time (rule 8, not re-hosted); scene assets stream at runtime into the mounted
# ~/.ai2thor cache.

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG AI2THOR_VERSION=4.0.0

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    HOME=/root \
    PYTHONPATH=/opt/sim:${PYTHONPATH}

WORKDIR /ryzers

# --- Headless Vulkan runtime for AI2-THOR CloudRendering -------------------------------------
# libvulkan1: Vulkan loader. mesa-vulkan-drivers: RADV ICD (radeon_icd.*.json). vulkan-tools:
# vulkaninfo for the Phase-2 render spike. libgl1/libglvnd0: Unity player GL deps. xvfb kept as a
# last-resort fallback backend (docs/ithor/PLAN.md ladder). On Ubuntu 22.04 the stock Mesa (23.2)
# predates RADV gfx1151 (RDNA3.5) support, so add the kisak-mesa PPA for a recent Mesa; on 24.04+
# the stock Mesa is new enough and the PPA step is skipped.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget curl unzip ca-certificates git \
        libvulkan1 mesa-vulkan-drivers vulkan-tools \
        libgl1 libglvnd0 libglu1-mesa libx11-6 xvfb && \
    . /etc/os-release && \
    if [ "${VERSION_ID%%.*}" -lt 24 ]; then \
        apt-get install -y --no-install-recommends software-properties-common gpg-agent && \
        add-apt-repository -y ppa:kisak/kisak-mesa && \
        apt-get update && \
        apt-get install -y --no-install-recommends mesa-vulkan-drivers ; \
    fi && \
    rm -rf /var/lib/apt/lists/*

# AI2-THOR python client (pure-python; the Unity build is fetched separately below). numpy pinned
# <2 to match the ai2thor 4.0.0 era + the AVDC image (mujoco/segmentation buffers overflow on
# numpy>=2), consistent across the chained stack. imageio+Pillow: the harness writes rollout videos.
RUN pip install --no-cache-dir "numpy==1.26.4" "ai2thor==${AI2THOR_VERSION}" \
      "imageio[ffmpeg]" "Pillow"

# The ai2thor Unity CloudRendering build (~540 MB) is NOT baked into the image on purpose: the
# runtime mounts a persistent volume over ~/.ai2thor (see config.yaml), which would shadow anything
# baked here - so build-time prefetch would only bloat the image while staying hidden at run time.
# Instead the first rollout lets the ai2thor Controller download the correct CloudRendering build
# (DEFAULT_CLOUDRENDERING_COMMIT_ID) into the mounted cache once (~15 s), reused thereafter (rule 8:
# fetched from the allenai public bucket, not re-hosted; rule 3: large asset stays on the remote).
# scripts/prefetch_build.sh (shipped below) can warm that cache explicitly if desired.

# Model-agnostic harness (importable as `sim_ithor`).
COPY lib/sim_ithor /opt/sim/sim_ithor

RUN python -c "import ai2thor, numpy; from ai2thor.platform import CloudRendering; \
from sim_ithor import Policy, load_policy; from sim_ithor.tasks import SCENE2TARGETS, get_cmat; \
print('OK sim-ithor import | ai2thor', ai2thor.__version__, '| numpy', numpy.__version__, \
'| tasks', sum(len(v) for v in SCENE2TARGETS.values()))"

COPY test.py /ryzers/
COPY scripts/ /ryzers/scripts/
COPY demos/ /ryzers/demos/
COPY examples/ /ryzers/examples/
RUN chmod +x /ryzers/scripts/*.sh /ryzers/demos/*.sh 2>/dev/null || true

# Point the Vulkan loader at the RADV ICD. On Mesa 25.x (Ubuntu 24.04) the ICD is named
# `radeon_icd.json`; older Mesa used the arch-suffixed `radeon_icd.x86_64.json`. Set the env to the
# canonical name and symlink the legacy name so either spelling (or an older base) resolves. Without
# this the loader finds no driver and the Unity player segfaults (returncode=-11) on CloudRendering.
ENV VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json
RUN d=/usr/share/vulkan/icd.d; \
    if [ -f "$d/radeon_icd.json" ] && [ ! -e "$d/radeon_icd.x86_64.json" ]; then \
        ln -s radeon_icd.json "$d/radeon_icd.x86_64.json"; \
    elif [ -f "$d/radeon_icd.x86_64.json" ] && [ ! -e "$d/radeon_icd.json" ]; then \
        ln -s radeon_icd.x86_64.json "$d/radeon_icd.json"; \
    fi

CMD ["python", "/ryzers/test.py"]
