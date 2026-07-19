# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Model-agnostic MimicGen (robosuite/MuJoCo) simulator base image for AMD Strix Halo
# (gfx1151, ROCm 7.2.2). Ships ONLY the simulator + a small harness (sim_mimicgen) that
# exposes a model-agnostic policy seam + a generic closed-loop / sanity runner. No policy or
# model weights: layer a VLA/WAM (VERA, ...) on top with `ryzers build simulation/mimicgen <model>`
# and select a policy at runtime via POLICY_FACTORY (see README).
#
# The sim/eval stack is VERA's upstream robosuite/robomimic/NVlabs-mimicgen/gym harness
# (`vera[eval]` + the MimicgenRunner + websocket client), reused unchanged (rule 2.1). We install
# ONLY the sim/eval half here (no diffusers/WAN/vggt/lightning), so this base stays lean and the
# planner/IDM only ever land in the model layer built on top. MimicGen task hdf5 are fetched at
# runtime into the mounted volume (rule 8), never baked.

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG VERA_COMMIT=fe8561aaa3783230d01d696987992f7eeacb2e12

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    PYTHONPATH=/opt/sim:${PYTHONPATH}

WORKDIR /ryzers

# git/cmake/build-essential/ffmpeg: build + video IO. GL *runtime* libs MuJoCo links against for
# headless offscreen rendering: EGL (libegl1/libgles2/libglvnd0), the Mesa GBM/DRI backend for EGL
# on the iGPU, OSMesa (libosmesa6) software fallback, GLFW + libGL for the robosuite viewer paths,
# and libglu1-mesa (silences the cosmetic libGLU.so.0 warning at env-close on gfx1151). The
# *-dev headers are needed because robomimic pulls `egl_probe`, which compiles a small EGL binary.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git cmake build-essential ffmpeg \
        libegl1 libgles2 libglvnd0 libgl1 libglx-mesa0 libgl1-mesa-dri libgbm1 \
        libosmesa6 libglfw3 libglew2.2 libglu1-mesa \
        libegl-dev libgles-dev libgl-dev mesa-common-dev && \
    rm -rf /var/lib/apt/lists/*

# Clone VERA (harness code: env_runner + controller + websocket client) at the pinned commit.
RUN git clone https://github.com/sizhe-li/VERA /repos/vera && \
    git -C /repos/vera checkout ${VERA_COMMIT}

# robomimic 0.5.0 is GitHub-only (PyPI stops at 0.3.0). It hard-pins an OLD diffusion/LM stack
# (transformers==4.41.2, diffusers==0.11.1, huggingface_hub==0.23.4) for its diffusion-policy /
# language-policy algorithms — none of which the MimicGen eval uses (robomimic is only the env
# wrapper + dataset utils here). We unpin exactly those three and install with --no-deps.
RUN git clone --depth 1 --branch v0.5.0 https://github.com/ARISE-Initiative/robomimic.git /repos/robomimic && \
    sed -i -E 's/(transformers|diffusers|huggingface_hub) *== *[0-9][0-9.]*/\1/' /repos/robomimic/setup.py && \
    grep -E "transformers|diffusers|huggingface_hub" /repos/robomimic/setup.py

# NVlabs mimicgen (EDITABLE, not the PyPI stub named "mimicgen" which lacks `mimicgen.utils`).
# The runner needs `mimicgen.utils.robomimic_utils.create_env`, which only exists in NVlabs/mimicgen.
# Editable keeps XML_ASSETS_BASE_PATH pointing into the clone where the per-task object XMLs +
# meshes live (~12 MB, NOT declared as package_data) — a `git+` build drops them and tasks that
# load a mimicgen object (e.g. coffee -> objects/coffee_pod.xml) crash with FileNotFoundError.
RUN git clone --depth 1 https://github.com/NVlabs/mimicgen.git /repos/mimicgen && \
    test -f /repos/mimicgen/mimicgen/models/robosuite/assets/objects/coffee_pod.xml

# Install the SIM/EVAL half of VERA only: strip the CUDA torch pin so the base ROCm torch is kept,
# unpin the mimicgen==1.0.0 stub, then install `vera[eval]` (gym/robomimic/robosuite/mimicgen/
# mujoco) — NOT [idm,video] (no diffusers/WAN/vggt here). The MimicgenRunner also imports a few
# light deps upstream buckets into other extras (opencv/h5py/scipy); add just those.
COPY scripts/strip_cuda_torch.py /ryzers/scripts/strip_cuda_torch.py
RUN python /ryzers/scripts/strip_cuda_torch.py /repos/vera/pyproject.toml && \
    sed -i -E 's/"mimicgen==1\.0\.0"/"mimicgen"/' /repos/vera/pyproject.toml && \
    grep -n "mimicgen" /repos/vera/pyproject.toml && \
    pip freeze | grep -iE "^(torch|torchvision|torchcodec)==" > /opt/vera-constraints.txt && \
    cat /opt/vera-constraints.txt && \
    PIP_CONSTRAINT=/opt/vera-constraints.txt pip install --no-cache-dir --no-deps -e /repos/robomimic && \
    PIP_CONSTRAINT=/opt/vera-constraints.txt pip install --no-cache-dir --no-deps -e /repos/mimicgen && \
    PIP_CONSTRAINT=/opt/vera-constraints.txt pip install --no-cache-dir gdown && \
    PIP_CONSTRAINT=/opt/vera-constraints.txt pip install --no-cache-dir -e "/repos/vera[eval]" && \
    PIP_CONSTRAINT=/opt/vera-constraints.txt pip install --no-cache-dir opencv-python h5py scipy

# Sim-side (client) port patch: disable the websocket-client keepalive (gfx1151 chunk stalls).
# Applied as a cheap late layer (vera is editable) so iterating never re-triggers the pip install.
COPY scripts/patch_sim_imports.py /ryzers/scripts/patch_sim_imports.py
RUN python /ryzers/scripts/patch_sim_imports.py /repos/vera

# Install the model-agnostic harness on PYTHONPATH (importable as `sim_mimicgen`).
COPY lib/sim_mimicgen /opt/sim/sim_mimicgen

RUN python -c "import torch; assert torch.version.hip, 'torch is not a ROCm build: '+torch.__version__; \
import robosuite, mimicgen, mujoco; from vera.env_runner.mimicgen_runner import MimicgenRunner; \
from sim_mimicgen import Policy, load_policy; print('OK sim-mimicgen import | torch', torch.__version__, 'hip', torch.version.hip)"

COPY test.py /ryzers/
COPY scripts/ /ryzers/scripts/
COPY demos/ /ryzers/demos/
COPY examples/ /ryzers/examples/
RUN chmod +x /ryzers/scripts/*.sh /ryzers/demos/*.sh 2>/dev/null || true

CMD ["python", "/ryzers/test.py"]
