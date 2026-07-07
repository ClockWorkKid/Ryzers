# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Model-agnostic LIBERO-Plus simulator base image for AMD Strix Halo (gfx1151).
# LIBERO-Plus (github.com/sylvestf/LIBERO-plus) is a drop-in replacement for the `libero`
# package that expands the four suites into 10,030 perturbation instances across 7
# robustness dimensions (camera, robot state, language, light, background, noise, layout)
# and 5 difficulty levels. This image ships that closed-loop stack (headless EGL) plus the
# sim_liberoplus harness (Policy seam + closed-loop/interactive/sanity runners and a
# built-in RandomPolicy). No policy/model code lives here: any policy chains on top of this
# image (ryzers build simulation/libero-plus <policy>) and is selected at runtime via
# POLICY_FACTORY. The ~6.4 GB perturbation asset pack is downloaded from the upstream
# HuggingFace dataset at build time (never re-hosted here); weights/datasets are supplied
# by the policy layer or mounted at runtime.

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_DISABLE_XET=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /ryzers

# LIBERO-Plus adds ImageMagick (wand) for texture/background perturbations on top of the
# LIBERO/MuJoCo/robosuite EGL render deps.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      git cmake build-essential unzip \
      libegl-dev libgles-dev libosmesa6-dev libglew-dev libglu1-mesa libglu1-mesa-dev \
      libexpat1 libfontconfig1-dev libpython3-stdlib libmagickwand-dev && \
    rm -rf /var/lib/apt/lists/*

# LIBERO-Plus closed-loop sim. numpy is pinned to 1.26.4 (robosuite's numba has no py3.12
# wheel for numpy 2.x); the constraint then protects the base ROCm torch + numpy so they are
# never replaced by transitive deps. LIBERO-Plus installs with --no-deps for the same reason.
# `future`, `cloudpickle`, `gym==0.25.2` and `termcolor` are LIBERO/robosuite runtime imports
# not declared as install deps; `wand` + `scikit-image` are the LIBERO-Plus extra_requirements
# (perturbation image ops). The `libero` package ships no top-level __init__.py, so it
# resolves via PYTHONPATH.
ENV PYTHONPATH=/opt/LIBERO-plus:/opt/sim
RUN pip install --no-cache-dir "numpy==1.26.4" && \
    pip freeze | grep -iE "^(torch|torchvision|torchcodec|numpy)==" > /opt/torch-constraints.txt && \
    PIP_CONSTRAINT=/opt/torch-constraints.txt pip install --no-cache-dir \
      "robosuite==1.4.0" "bddl==1.0.1" "mujoco==3.3.2" easydict thop \
      "future==1.0.0" "cloudpickle==3.1.2" "gym==0.25.2" termcolor \
      "imageio>=2.34" "imageio-ffmpeg>=0.5" "pillow>=10" \
      "wand" "scikit-image" "huggingface_hub>=0.24" && \
    git clone --depth 1 https://github.com/sylvestf/LIBERO-plus /opt/LIBERO-plus && \
    PIP_CONSTRAINT=/opt/torch-constraints.txt pip install --no-cache-dir --no-deps -e /opt/LIBERO-plus

# Perturbation asset pack (~6.4 GB: hundreds of objects, textures, scenes) from the upstream
# open-source HuggingFace dataset. Never re-hosted in this repo (compliance). Kept as the sole
# action of its own layer (only the download command hashes into the cache key) so the slow
# fetch is reused across rebuilds even when the relocate/unpack logic below is edited.
RUN python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='Sylvest/LIBERO-plus', repo_type='dataset', filename='assets.zip', local_dir='/opt/lpdl')" && \
    rm -rf /root/.cache/huggingface

# Unpack + relocate. The zip preserves the uploader's absolute directory prefix
# (inspire/hdd/.../LIBERO-plus-0/assets), so we extract to a scratch dir, locate the real
# `assets` root (the parent of the committed `assets/scenes`), and move it to where the
# `libero` package expects it (libero/libero/assets). mv is a same-filesystem rename during
# the build so the 9.5 GB move is instant. The final test asserts a known base scene XML
# resolves at the expected path. The zip is dropped so it is not baked into the image.
RUN unzip -q -o /opt/lpdl/assets.zip -d /tmp/lpx && \
    ASRC="$(dirname "$(find /tmp/lpx -type d -path '*/assets/scenes' | head -1)")" && \
    test -d "$ASRC" && \
    mkdir -p /opt/LIBERO-plus/libero/libero && \
    rm -rf /opt/LIBERO-plus/libero/libero/assets && \
    mv "$ASRC" /opt/LIBERO-plus/libero/libero/assets && \
    rm -rf /tmp/lpx /opt/lpdl && \
    test -f /opt/LIBERO-plus/libero/libero/assets/scenes/libero_tabletop_base_style.xml

# The sim harness (Policy seam, closed-loop/interactive/sanity runners, RandomPolicy).
COPY lib/ /opt/sim/

# Install the torch.load compat shim as a startup .pth in site-packages so it applies to
# every Python process in the image (LIBERO-Plus loads pickled init-states via torch.load,
# which py>=2.6 rejects under weights_only=True). See _torch_compat.
RUN cp /opt/sim/sim_liberoplus_compat.pth "$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/"

# First `import libero.libero` writes ~/.libero/config.yaml interactively; feed defaults so it
# is baked into the image and never prompts at runtime. Then verify the deeper env import
# (robosuite controllers), the sim_liberoplus harness, and that the perturbation
# classification (task_classification.json) resolves. No GPU render here (build has no
# device); the sanity demo exercises rendering at run time.
RUN printf 'N\nN\nN\nN\nN\n' | python -c "import mujoco, robosuite; from libero.libero import benchmark; from libero.libero.envs import OffScreenRenderEnv; import sim_liberoplus; from sim_liberoplus.policy import load_policy; from sim_liberoplus.libero_env import load_task_classification; c = load_task_classification(); print('OK mujoco', mujoco.__version__, '| suites in classification:', list(c.keys()))"

COPY test.py /ryzers/
COPY scripts/ /ryzers/scripts/
COPY demos/ /ryzers/demos/
RUN chmod +x /ryzers/demos/*.sh /ryzers/scripts/*.sh

CMD ["python", "/ryzers/test.py"]
