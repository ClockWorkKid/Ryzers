
![](docs/header.png)

# Ryzen AI Robotics, Vision and ML Dockerfiles

This repository provides a collection of composable Dockerfiles and build scripts for deploying and running software, full applications, and select demonstrators on AMD Ryzen AI hardware. The project is designed to streamline the setup of AI workloads, robotics, vision, and other applications optimized for Ryzen AI.

---

## Overview

Ryzers is a modular framework for building and running Docker containers tailored for AMD Ryzen AI hardware. It supports a wide range of applications, including machine learning, robotics, vision, and more. The repository is structured to allow easy composition of Dockerfiles, enabling users to build custom containers for their specific needs.

These dockerfiles will also be pushed and actively maintained in their original repository homes whenever possible.  Ryzers will be a collection point of software frameworks to run on AMD hardware.  We are committed to open-source and happy to accept contributions or feedback on packages hosted here.

---

## Supported Packages

| Category        | Software                                                                                                    |
|-----------------|--------------------------------------------------------------------------------------------------------------------|
| LLM                     | [`ollama`](packages/llm/ollama), [`llamacpp`](packages/llm/llamacpp), [`lmstudio`](packages/llm/lmstudio), [`GEPA`](packages/llm/gepa), [`Lemonade`](packages/llm/lemonade-sdk)   |
| VLM                     | [`Gemma3`](packages/vlm/gemma3), [`SmolVLM`](packages/vlm/smolvlm), [`Phi-4`](packages/vlm/phi4), [`LFM2-VL`](packages/vlm/lfm2vl) |
| VLA                     | [`OpenVLA`](packages/vla/openvla), [`SmolVLA`](packages/vla/smolvla), [`GR00T-N1.5`](packages/vla/gr00t), [`openpi`](packages/vla/openpi), [`CogACT`](packages/vla/cogact), [`MolmoAct`](packages/vla/molmoact), [`MolmoAct2`](packages/vla/molmoact2), [`VLA-JEPA`](packages/vla/vlajepa) |
| WAM                     | [`FastWAM`](packages/wam/fastwam), [`AHA-WAM`](packages/wam/ahawam), [`X-WAM`](packages/wam/xwam), [`VERA`](packages/wam/vera) |
| Graphics                     | [`O3DE`](packages/graphics/o3de) |
| Robotics                | [`ROS 2`](packages/ros/ros), [`Gazebo`](packages/ros/gazebo), [`LeRobot`](packages/robotics/lerobot), [`ACT`](packages/robotics/act), [`RAI`](packages/robotics/rai)    |
| Simulation                |  [`Genesis`](packages/robotics/genesis), [`PyDrake`](packages/robotics/pydrake), [`LIBERO`](packages/simulation/libero), [`LIBERO-Plus`](packages/simulation/libero-plus), [`RoboTwin`](packages/simulation/robotwin), [`RoboCasa`](packages/simulation/robocasa), [`SimplerEnv`](packages/simulation/simplerenv)  |
| Vision                  | [`OpenCV`](packages/vision/opencv), [`SAM`](packages/vision/sam), [`MobileSAM`](packages/vision/mobilesam), [`ncnn`](packages/vision/ncnn), [`DINOv3`](packages/vision/dinov3), [`SAM3`](packages/vision/sam3), [`Ultralytics`](packages/vision/ultralytics) |
| Ryzen AI NPU                |  [`XDNA`](packages/npu/xdna), [`IRON`](packages/npu/iron), [`NPUEval`](packages/npu/npueval), [`Ryzen AI CVML`](packages/npu/ryzenai_cvml)  |
| Adaptive SoCs           | [`PYNQ.remote`](packages/adaptive-socs/pynq-remote) |
| Utilities   | [`JupyterLab`](packages/ide/jupyterlab), [`amdgpu_top`](packages/init/amdgpu_top) |

---

## Simulation benchmark bases

Closed-loop robot-manipulation benchmarks are packaged as **model-agnostic simulator base
images** under `packages/simulation/`. Each ships a self-contained harness (closed-loop /
interactive / sanity runners) and a `Policy` seam: a model layers on top with
`ryzers build <sim-base> <model>` and plugs in through a runtime adapter selected by the
`POLICY_FACTORY` environment variable. This keeps a single source of truth per simulator so
every model consumes the same base (see [`packages/wam/fastwam`](packages/wam/fastwam) for the
reference adapter pattern).

| Simulator base | Backend | Description | Consumed by |
|----------------|---------|-------------|-------------|
| [`simulation/libero`](packages/simulation/libero) | MuJoCo | LIBERO manipulation suites (`libero_object/goal/spatial/10/90`) with interactive + closed-loop harness. | `wam/fastwam`, `vla/vlajepa`, `vla/molmoact2` |
| [`simulation/libero-plus`](packages/simulation/libero-plus) | MuJoCo | LIBERO-Plus robustness benchmark: the four suites expanded into 10,030 perturbation instances across 7 dimensions × 5 difficulty levels. Harness synced to the `simulation/libero` base. | `vla/vlajepa` |
| [`simulation/robotwin`](packages/simulation/robotwin) | SAPIEN / Vulkan | RoboTwin 2.0 dual-arm manipulation (PyTorch3D on ROCm/gfx1151). | `wam/fastwam`, `wam/ahawam`, `wam/xwam` |
| [`simulation/robocasa`](packages/simulation/robocasa) | MuJoCo | RoboCasa kitchen manipulation (24 tasks, RoboSuite v1.5, OSC_POSE single-arm) with interactive + closed-loop harness. | `wam/xwam` |
| [`simulation/simplerenv`](packages/simulation/simplerenv) | SAPIEN3 / ManiSkill3 | SimplerEnv real-to-sim (Google Robot + WidowX/Bridge), CPU physics. **Under development** — scaffold + `Policy` seam in place; gfx1151 render smoke pending. | `vla/vlajepa` (planned) |

---

## Installation

To get started, clone the repository and install the required dependencies:

```bash
git clone https://github.com/AMDResearch/Ryzers
pip install Ryzers/
```

For detailed installation instructions and requirements, refer to the [included documentation](https://amdresearch.github.io/Ryzers/installation.html).

## Usage


### Simple Example
```
ryzers build genesis
ryzers run 
```

```
# Alternatively, override the Dockerfile's CMD to run a custom command
ryzers run bash
```

For detailed build and run instructions, refer to the instructions included in the package.

## Features

- **Verified AMD Support for Popular Frameworks**: A variety of Robotics and ML software frameworks supported across Ryzen AI platforms.
- **Optimized for Ryzen AI**: Includes support for hardware-accelerated AI workloads and accelerators like the iGPU and NPUs.
- **Minimal Host Software Requirements**: Standard Ubuntu support with minimal software requirements
- **Composable Dockerfiles**: Modular design for reusability across different applications.

---

## Contributing

We welcome contributions to Ryzers! If you have ideas for new features, bug fixes, or improvements, please submit a pull request or open an issue.

**Using an AI coding agent?** Point it at [`skills/new-ryzer/SKILL.md`](skills/new-ryzer/SKILL.md) — it has everything an agent needs to scaffold a new package correctly (templates, conventions, config schema, and reference examples). Works with any agent: Claude Code, Codex, Cursor, Copilot, etc.

For detailed guidelines, see [CONTRIBUTING.md](docs/contributing.md).

---

## License

This project is licensed under the MIT license. See the [`LICENSE`](LICENSE) file for details. 
