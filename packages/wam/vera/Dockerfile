# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# VERA (Video-to-Embodied Robot Action model) policy/model layer for AMD Strix Halo
# (gfx1151). Direct PyTorch -> ROCm port: installs VERA's `.[idm,video]` stages on the base
# image's ROCm torch (the upstream `torch==2.6.0` CUDA pin is stripped so the base ROCm torch
# is preserved). flash-attn is intentionally NOT installed -- the WAN planner falls back to
# SDPA/AOTriton on ROCm. VGGT (the IDM visual backbone) installs from its public git dep.
#
# This is a planner+IDM+sim image. `.[idm,video]` cover the video-generation path; `.[eval]`
# adds the MimicGen/PushT MuJoCo stacks (robosuite/robomimic/mimicgen/gym-pusht) for closed-loop
# demos, plus the EGL/OSMesa GL libs MuJoCo needs to render offscreen (headless, no X11). Weights
# and the frozen upstream bases (Wan2.1 T2V/I2V, VGGT-1B) are fetched at runtime into the mounted
# HF cache (rule 8).

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG VERA_COMMIT=fe8561aaa3783230d01d696987992f7eeacb2e12

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /ryzers

# git/cmake/build-essential/ffmpeg: build + video IO. The GL *runtime* libs MuJoCo links
# against for headless offscreen rendering: EGL (libegl1/libgles2/libglvnd0), the Mesa
# GBM/DRI backend for EGL on the iGPU, OSMesa (libosmesa6) software fallback, GLFW + libGL
# for the robosuite viewer paths. The *-dev headers (libegl-dev/libgles-dev/mesa-common-dev)
# are needed because robomimic pulls `egl_probe`, which compiles a small EGL query binary.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git cmake build-essential ffmpeg \
        libegl1 libgles2 libglvnd0 libgl1 libglx-mesa0 libgl1-mesa-dri libgbm1 \
        libosmesa6 libglfw3 libglew2.2 \
        libegl-dev libgles-dev libgl-dev mesa-common-dev && \
    rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/sizhe-li/VERA /repos/vera && \
    git -C /repos/vera checkout ${VERA_COMMIT}

# Pin the base's torch + torchvision so the VERA install (and its extras) never replaces
# them with CUDA wheels. The upstream `torch==2.6.0` pin (ABI-tied to flash-attn CUDA wheels)
# is removed; everything else stays faithful to the pinned upstream.
# NOTE: numpy is intentionally NOT pinned here. The IDM backbone `vggt` hard-requires
# numpy<2, which conflicts with the base image's numpy 2.4.4. ROCm torch 2.10 + torchvision
# 0.25 import and round-trip cleanly under numpy 1.26.4 (probed on gfx1151), so we let the
# resolver settle numpy at vggt's <2 cap instead of forcing the base's 2.x.
COPY scripts/strip_cuda_torch.py /ryzers/scripts/strip_cuda_torch.py
# robomimic 0.5.0 is GitHub-only (PyPI stops at 0.3.0); install it from the pinned v0.5.0
# tag first so the `.[eval]` extra's `robomimic==0.5.0` is already satisfied. robomimic
# hard-pins an OLD diffusion/LM stack (transformers==4.41.2, diffusers==0.11.1,
# huggingface_hub==0.23.4) for its diffusion-policy / language-policy algorithms — none of
# which VERA's PushT/MimicGen eval uses (robomimic is only the env wrapper + dataset utils
# here). Those pins clash head-on with the WAN planner (diffusers>=0.34, transformers>=4.51),
# so we unpin exactly those three in robomimic's setup.py and install with --no-deps; the
# `.[idm,video,eval]` resolve then settles the shared stack at the WAN-compatible versions.
RUN git clone --depth 1 --branch v0.5.0 https://github.com/ARISE-Initiative/robomimic.git /repos/robomimic && \
    sed -i -E 's/(transformers|diffusers|huggingface_hub) *== *[0-9][0-9.]*/\1/' /repos/robomimic/setup.py && \
    grep -E "transformers|diffusers|huggingface_hub" /repos/robomimic/setup.py
# The eval extra pins `mimicgen==1.0.0`, but the PyPI project named "mimicgen" is an unrelated
# stub (ships __version__ 0.1.0, no `mimicgen.utils`) — VERA's MimicGen runner needs
# `mimicgen.utils.robomimic_utils.create_env`, which only exists in NVlabs/mimicgen (GitHub-only,
# reports 1.0.1). So we unpin mimicgen in the extra and install NVlabs/mimicgen from source with
# --no-deps (its setup.py hard-pins old robomimic/robosuite/mujoco we don't want); `gdown` is the
# one runtime import its file_utils needs. The unpinned `mimicgen` req is then satisfied by 1.0.1.
#
# We clone + install EDITABLE (`-e /repos/mimicgen`) rather than the `git+` form, because
# mimicgen's per-task object XMLs + meshes (mimicgen/models/robosuite/assets/, ~12 MB, in-tree)
# are NOT declared as package_data, so a `git+` build drops them and tasks that load a
# mimicgen-specific object (e.g. coffee -> objects/coffee_pod.xml) crash with FileNotFoundError.
# Editable keeps XML_ASSETS_BASE_PATH pointing into the clone where the assets live (rule 2.1).
RUN git clone --depth 1 https://github.com/NVlabs/mimicgen.git /repos/mimicgen && \
    test -f /repos/mimicgen/mimicgen/models/robosuite/assets/objects/coffee_pod.xml
RUN python /ryzers/scripts/strip_cuda_torch.py /repos/vera/pyproject.toml && \
    sed -i -E 's/"mimicgen==1\.0\.0"/"mimicgen"/' /repos/vera/pyproject.toml && \
    grep -n "mimicgen" /repos/vera/pyproject.toml && \
    pip freeze | grep -iE "^(torch|torchvision|torchcodec)==" > /opt/vera-constraints.txt && \
    cat /opt/vera-constraints.txt && \
    PIP_CONSTRAINT=/opt/vera-constraints.txt pip install --no-cache-dir --no-deps -e /repos/robomimic && \
    PIP_CONSTRAINT=/opt/vera-constraints.txt pip install --no-cache-dir --no-deps -e /repos/mimicgen && \
    PIP_CONSTRAINT=/opt/vera-constraints.txt pip install --no-cache-dir gdown && \
    PIP_CONSTRAINT=/opt/vera-constraints.txt pip install --no-cache-dir -e "/repos/vera[idm,video,eval]"

# VERA source patches, applied AFTER the (expensive, cached) dependency install. vera is an
# editable install (`-e`), so rewriting the checked-out source takes effect at import time. Kept
# as its own late layer so iterating on the patch never re-triggers the pip reinstall above:
#   * torchmetrics 1.4.x LPIPS import guard (eval-only metrics, off the inference path)
#   * tracker_backend_from_cfg: honor MotionTrackConfig's `backend` field so the MimicGen
#     algo_config's `tracker.backend: cotracker` actually selects cotracker over alltracker.
COPY scripts/patch_eval_imports.py /ryzers/scripts/patch_eval_imports.py
RUN python /ryzers/scripts/patch_eval_imports.py /repos/vera

RUN python -c "import torch; assert torch.version.hip, 'torch is not a ROCm build: '+torch.__version__; import vera, vera.policy, vera.idm, vera.server, vera.controller; print('OK vera on torch', torch.__version__, 'hip', torch.version.hip)"

COPY test.py test_eval.py /ryzers/
COPY scripts/ /ryzers/scripts/
COPY demos/ /ryzers/demos/
RUN chmod +x /ryzers/scripts/*.sh /ryzers/demos/*.sh 2>/dev/null || true

CMD ["python", "/ryzers/test.py"]
