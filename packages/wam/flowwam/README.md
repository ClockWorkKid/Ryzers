### FlowWAM

This package runs [FlowWAM](https://github.com/YixiangChen515/FlowWAM) on AMD Ryzen AI Max+ 395 (Strix Halo, `gfx1151`) under ROCm 7.14. FlowWAM is a Wan2.2-TI2V-5B dual-stream world-action model that uses dense optical flow as a unified action representation: a dual-stream Wan DiT jointly generates a future RGB video and a flow field, an IDM action expert decodes a 14-DoF robot action chunk from that flow, and RoboTwin 2.0 embodiments are rendered headless via SAPIEN. It ships no simulator, so it chains on the RoboTwin sim base for closed-loop control.

Upstream is a trimmed DiffSynth (the same Wan-video framework FastWAM builds on), so this reuses FastWAM's ROCm playbook: strip the base-owned CUDA/torch/numpy/sapien pins, hold them via `PIP_CONSTRAINT`, and install upstream on the base image's ROCm torch. No flash-attn/apex/cuBLAS are added -- DiffSynth's `flash_attention()` falls back to torch SDPA on ROCm.

### Build

FlowWAM requires SAPIEN robot rendering + RoboTwin embodiments, so chain it on the RoboTwin 2.0 base (Vulkan SAPIEN, headless on gfx1151):

```sh
ryzers build robotwin flowwam --name flowwam-robotwin   # chain the model on the RoboTwin base
ryzers run   --name flowwam-robotwin                     # test.py: ROCm torch + GPU + Wan dual-stream deps
```

Artifacts are written to `workspace/flowwam/outputs`. Set `HF_TOKEN` for faster/gated downloads. The Wan2.2-TI2V-5B base (~24 GB) + FlowWAM RoboTwin checkpoint are fetched on the first closed-loop run, or explicitly:

```sh
ryzers run --name flowwam-robotwin /ryzers/scripts/download_checkpoints.sh base      # Wan2.2-TI2V-5B + UMT5-XXL enc
ryzers run --name flowwam-robotwin /ryzers/scripts/download_checkpoints.sh robotwin  # FlowWAM action policy ckpt
```

### Smoke test

Env sign-of-life (no weights): proves ROCm torch + the DiffSynth Wan dual-stream pipeline + SAPIEN/flow deps import and the SDPA attention fallback runs.

```sh
ryzers run --name flowwam-robotwin /ryzers/demos/demo_smoke.sh
```

### Closed-loop RoboTwin 2.0

Genuine control loop: the dual-stream DiT generates flow-conditioned video latents, the IDM action expert decodes a 14-DoF action chunk, and RoboTwin steps the SAPIEN env, replanning every `EXECUTE_WINDOW` actions until success or timeout. RoboTwin scores task success and writes a per-episode video.

```sh
ryzers run --name flowwam-robotwin /ryzers/demos/demo_closedloop_robotwin.sh            # default: beat_block_hammer
TASK=beat_block_hammer NUM_EPISODES=3 \
  ryzers run --name flowwam-robotwin /ryzers/demos/demo_closedloop_robotwin.sh
```

The dual-stream Video DiT denoises on the gfx1151 iGPU (bf16 SDPA, no flash-attn), and the
`beat_block_hammer` task is solved with a 14-DoF action chunk replanned every 25 steps.

<p align="center">
  <img src="assets/closedloop_robotwin_beat_block_hammer.gif" width="420">
  <br><em>Closed-loop RoboTwin 2.0 <code>beat_block_hammer</code> rollout on Strix Halo (gfx1151),
  rendered headless via SAPIEN (verified 1/1 success).</em>
</p>

### References

- Upstream (dual-stream world-action policy): https://github.com/YixiangChen515/FlowWAM
- World-model variant: https://github.com/YixiangChen515/FlowWAM_WorldArena
- Model + checkpoints: https://huggingface.co/YixiangChen/FlowWAM
- Base model: https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B

Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
