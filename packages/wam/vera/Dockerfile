# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# VERA (Video-to-Embodied Robot Action model) — the MODEL layer (WAN video planner + Jacobian
# IDM) for AMD Strix Halo (gfx1151), built ON TOP of the model-agnostic MimicGen simulator base:
#
#     ryzers build simulation/mimicgen vera --name vera
#
# The sim base (packages/simulation/mimicgen) already provides the robosuite/robomimic/NVlabs-
# mimicgen/MuJoCo stack + GL libs + the cloned VERA repo (`/repos/vera`, editable, with the CUDA
# torch pin stripped and `vera[eval]` installed) + the sim_mimicgen harness. Here we add ONLY the
# model half: `vera[idm,video]` (the WAN Wan2.1 DiT planner + VGGT-1B Jacobian IDM), the policy
# server, the model-side gfx1151 patches + kernel speedups, and the DROID video-gen + PushT paths.
# flash-attn is intentionally NOT installed (WAN falls back to SDPA/AOTriton on ROCm). Weights and
# the frozen upstream bases (Wan2.1 T2V/I2V, VGGT-1B) are fetched at runtime (rule 8).
#
# The MimicGen closed-loop is now split across the two images: this model image serves the WAN+IDM
# policy over the websocket protocol, and the sim base's sim_mimicgen harness drives the robosuite
# env against it (demos/demo_closedloop_mimicgen.sh). PushT + DROID stay wholly in this image.

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG VERA_COMMIT=fe8561aaa3783230d01d696987992f7eeacb2e12

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /ryzers

# Sanity: this MUST be built on the simulation/mimicgen base (it supplies /repos/vera + sim stack).
RUN test -d /repos/vera -a -f /opt/vera-constraints.txt || { \
      echo "ERROR: build FROM the simulation/mimicgen base: ryzers build simulation/mimicgen vera"; \
      exit 1; }

# Add the MODEL half on top of the base's editable `vera[eval]`: `[idm,video]` pulls the WAN
# planner (diffusers/transformers/pytorch-lightning/deepspeed) + the VGGT-1B Jacobian IDM (vggt
# git dep, scipy/h5py/transforms3d/rerun/…). The base's torch constraint keeps ROCm torch pinned so
# none of these swap in a CUDA wheel. numpy stays unpinned so vggt's numpy<2 cap round-trips.
RUN PIP_CONSTRAINT=/opt/vera-constraints.txt pip install --no-cache-dir -e "/repos/vera[idm,video]"

# Model-side VERA source patches (applied to the shared editable clone; vera is `-e`, so rewriting
# the source takes effect at import). Kept as cheap late layers so iterating never re-triggers the
# pip install above. patch_eval_imports covers the eval-metric LPIPS guards, the cotracker tracker-
# backend + visualization fixes, and the server-side keepalive (the client-side keepalive was
# already applied by the sim base; that one line no-ops here). patch_gfx1151_speedups bakes the two
# validated kernel-level wins (conv override + bf16 IDM autocast) into start_server_mimicgen.
COPY scripts/patch_eval_imports.py /ryzers/scripts/patch_eval_imports.py
COPY scripts/patch_gfx1151_speedups.py /ryzers/scripts/patch_gfx1151_speedups.py
RUN python /ryzers/scripts/patch_eval_imports.py /repos/vera && \
    python /ryzers/scripts/patch_gfx1151_speedups.py /repos/vera

RUN python -c "import torch; assert torch.version.hip, 'torch is not a ROCm build: '+torch.__version__; \
import vera, vera.policy, vera.idm, vera.server, vera.controller; \
import sim_mimicgen; \
print('OK vera model layer on torch', torch.__version__, 'hip', torch.version.hip)"

COPY test.py test_eval.py /ryzers/
COPY scripts/ /ryzers/scripts/
COPY demos/ /ryzers/demos/
COPY adapters/ /opt/vera-adapters/
COPY RUNTIME_OPTIMIZATION.md /ryzers/RUNTIME_OPTIMIZATION.md
# The VERA MimicGen adapter is a thin sim_mimicgen.Policy factory; put it on PYTHONPATH so
# POLICY_FACTORY=vera_mimicgen_policy:build_policy resolves when driving the sim base harness.
ENV PYTHONPATH=/opt/vera-adapters:${PYTHONPATH}
RUN chmod +x /ryzers/scripts/*.sh /ryzers/demos/*.sh 2>/dev/null || true

CMD ["python", "/ryzers/test.py"]
