"""Build-time sanity check (COPY'd + run; legacy docker builder mangles heredocs).

No GPU at build: verifies torch stays ROCm, torchdistill registries populate, the
student produces the seam shape, and the closed-loop LIBERO sim stack imports.
"""

import torch

assert torch.version.hip, "torch is not ROCm: " + torch.__version__

import torchdistill
import transformers

import distill.register  # registers teacher/student/loss
from torchdistill.models.registry import get_model

student = get_model("vit_distill_student", None, variant="hybrid")
seam = student(torch.randn(1, 2, 729, 588))
assert tuple(seam.shape) == (1, 2, 729, 2304), seam.shape

import bddl
import mujoco
import robosuite
from libero.libero import benchmark

print("torch", torch.__version__, "| hip", torch.version.hip)
print("torchdistill", torchdistill.__version__, "| transformers", transformers.__version__)
print("robosuite", robosuite.__version__, "| mujoco", mujoco.__version__)
print("LIBERO suites:", sorted(benchmark.get_benchmark_dict().keys()))
print("student seam ok:", tuple(seam.shape))
