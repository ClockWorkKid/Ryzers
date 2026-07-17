# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Nano World Model (NanoWM) — diffusion-forcing video world model — for AMD Strix Halo
# (gfx1151). This is a slim world-model image: it installs NanoWM's upstream code on the
# base image's ROCm torch (upstream is conda/CUDA; we drop conda + pytorch-cuda and keep the
# base ROCm torch, installing the pip deps under a constraint that pins base torch+numpy).
# It ships NO simulator; the rollout/video-gen path runs on the plain ROCm base. Weights,
# datasets and codec weights are fetched at runtime into a mounted HF cache (rule 8).
#
# Phase-1 validated on strix-halo (gfx1151, ROCm 7.2.2, base torch 2.10 / py3.12):
#  * pytorch-lightning==1.9.5 installs cleanly on the base — kept at upstream's pin.
#  * diffusers: upstream pins ==0.24.0, but that imports huggingface_hub.constants.hf_cache_home
#    which the base image's newer huggingface_hub removed -> ImportError. NanoWM only uses
#    diffusers for AutoencoderKL, get_scheduler and is_{bs4,ftfy}_available (stable APIs; it
#    implements its own diffusion), so we unpin diffusers to match the base hub (rule 2.1 patch).

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG NANOWM_COMMIT=2ee3c355a6e44d2254eba8a8c29f199adddf4a3e

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /ryzers

RUN apt-get update && \
    apt-get install -y --no-install-recommends git cmake build-essential ffmpeg && \
    rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/simchowitzlabpublic/nano-world-model /repos/nanowm && \
    git -C /repos/nanowm checkout ${NANOWM_COMMIT}

# ROCm/project-specific patches applied on top of the pinned upstream (rule 2.1: minimal, and
# only what upstream is missing/broken for our path). See patches/ for the rationale per file.
COPY patches/ /tmp/nanowm-patches/
RUN for p in /tmp/nanowm-patches/*.patch; do echo "Applying $p"; git -C /repos/nanowm apply "$p"; done

# Pin the base's torch + torchvision + numpy so the NanoWM install never replaces them with a
# CUDA build. environment.yml is conda-based (pytorch-cuda + nvidia channel); we translate its
# `pip:` section into a pip install and drop the conda torch path entirely.
RUN pip freeze | grep -iE "^(torch|torchvision|numpy)==" > /opt/nanowm-constraints.txt && \
    PIP_CONSTRAINT=/opt/nanowm-constraints.txt pip install --no-cache-dir \
      "scipy" "Pillow" "tqdm" "requests" \
      "timm" "einops" "lpips" "piqa" "torchmetrics" \
      "diffusers" "accelerate" "transformers" \
      "omegaconf" "hydra-core" "tensorboard" "wandb" \
      "decord" "h5py" "pandas" \
      "opencv-python-headless" "imageio" "imageio-ffmpeg" "moviepy==1.0.3" "scikit-image" "av" \
      "pytorch-fid" "matplotlib" "seaborn" "sentencepiece" "beautifulsoup4" "ftfy" "protobuf==4.25.3"

# pytorch-lightning 1.9.5 (upstream's pin) installs cleanly on the base. (lerobot for RT-1 is
# baked as a late layer below.) Its lightning_fabric calls pkg_resources.declare_namespace, but the base venv's
# setuptools (82) no longer ships pkg_resources -> ModuleNotFoundError on `import experiments`
# (training/eval-via-main.py path). Restore pkg_resources with a setuptools that still vendors
# it (rule 2.1 patch; the rollout/video-gen path does not need Lightning at all).
RUN PIP_CONSTRAINT=/opt/nanowm-constraints.txt pip install --no-cache-dir \
      "pytorch-lightning==1.9.5" "setuptools<81"

RUN python -c "import torch; assert torch.version.hip, 'torch is not a ROCm build: '+torch.__version__; print('OK torch', torch.__version__, 'hip', torch.version.hip)"

# --- Demo/application extras (added as late, separately-cached layers) ---
# zip/unzip: recombine the DINO-WM deformable split archive (rope+granular).
RUN apt-get update && apt-get install -y --no-install-recommends zip unzip && \
    rm -rf /var/lib/apt/lists/*

# Always-on demo deps: PyYAML (config parsing in demo scripts), ffmpeg already present.
RUN PIP_CONSTRAINT=/opt/nanowm-constraints.txt pip install --no-cache-dir pyyaml

# Planning application (PushT sim): gym + pymunk + pygame + shapely. point_maze also needs
# MuJoCo 2.10 + d4rl (heavy binary) which we intentionally skip; pusht planning is the demo.
# Soft-fail so a resolver hiccup on py3.12 never breaks the core world-model image.
# pymunk<7: the PushT env uses space.add_collision_handler(), removed in pymunk 7.x.
RUN PIP_CONSTRAINT=/opt/nanowm-constraints.txt pip install --no-cache-dir \
      "gym==0.26.2" "pymunk==6.8.1" pygame shapely || echo "WARN: planning sim deps failed (planning demo best-effort)"

# video->3D application: Depth Anything 3 + viser. DA3 model weights download at runtime (rule 8).
# Soft-fail: this is an optional application layer.
RUN PIP_CONSTRAINT=/opt/nanowm-constraints.txt pip install --no-cache-dir viser || echo "WARN: viser failed"
# DA3 upstream pins numpy<2, which conflicts with the base ROCm torch (numpy 2.x). Install DA3
# itself with --no-deps (keeps base torch/numpy), then add its actual runtime deps under the
# base constraint. numpy 2 is API-compatible for DA3's inference path in practice.
RUN git clone --depth 1 https://github.com/ByteDance-Seed/Depth-Anything-3 /repos/da3 && \
    PIP_CONSTRAINT=/opt/nanowm-constraints.txt pip install --no-cache-dir --no-deps -e /repos/da3 && \
    PIP_CONSTRAINT=/opt/nanowm-constraints.txt pip install --no-cache-dir \
      addict e3nn evo fastapi uvicorn typer plyfile pillow_heif trimesh pycolmap rich safetensors \
      || echo "WARN: Depth-Anything-3 install failed (video_to_3d demo best-effort)"

# RT-1 application: LeRobot v2.0 dataset loading. lerobot 0.3.3 is the version compatible with the
# fractal v2.0 mirror (IPEC-COMMUNITY/fractal20220817_data_lerobot); newer lerobot (0.6.x) drops
# v2.0 support. Its declared deps pin a CUDA torch (<2.8) + torchcodec, so install --no-deps to keep
# the base ROCm torch, then add only the non-torch runtime deps. Video decodes via pyav (`av`, in the
# base layer) because torchcodec is absent and lerobot auto-falls back to it. The RT-1 video-gen path
# needs only dataset loading, so robot-HW/viz extras (pynput/pyserial/rerun/flask/cmake) are skipped.
RUN PIP_CONSTRAINT=/opt/nanowm-constraints.txt pip install --no-cache-dir --no-deps "lerobot==0.3.3" && \
    PIP_CONSTRAINT=/opt/nanowm-constraints.txt pip install --no-cache-dir \
      "datasets<=3.6.0" "draccus==0.10.0" "gymnasium==0.29.1" \
      jsonlines "deepdiff<9.0.0" orderly-set pyarrow dill multiprocess xxhash aiohttp \
      mergedeep pyyaml-include typing-inspect mypy_extensions termcolor toml farama-notifications \
      || echo "WARN: lerobot (RT-1) install failed (rt1 demo best-effort)"
RUN python -c "import torch; assert torch.version.hip, 'torch clobbered by lerobot deps: '+torch.__version__; print('OK torch still ROCm', torch.__version__)"

ENV PYTHONPATH=/repos/nanowm:/repos/nanowm/src

COPY test.py /ryzers/
COPY scripts/ /ryzers/scripts/
COPY demos/ /ryzers/demos/
RUN chmod +x /ryzers/scripts/*.sh /ryzers/demos/*.sh 2>/dev/null || true

CMD ["python", "/ryzers/test.py"]
