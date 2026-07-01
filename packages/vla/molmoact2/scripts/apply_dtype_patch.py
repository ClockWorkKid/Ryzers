#!/usr/bin/env python3
"""Idempotently patch the installed lerobot MolmoAct2 policy so the HF model is loaded
in a configurable dtype (env MOLMOACT2_DTYPE, default bfloat16) instead of the hardcoded
float32.

Why: the lerobot wrapper (lerobot/policies/molmoact2/modeling_molmoact2.py) loads with
`dtype=torch.float32`, which on Strix Halo (gfx1151) runs the LLM/vision matmuls ~5x
slower than the validated bf16 inference path used by the DROID benchmark (~3.2 s vs
~0.6 s per plan). bf16 is MolmoAct2's intended inference precision (the upstream
host_server loader is bf16), so this both matches deployment and unblocks real-time
chunk stitching. Not auto-applied at build time (so the validated fp32 closed-loop
eval path is unchanged); invoked by the launchers that want deployment-speed inference.
Safe to re-run."""
import importlib.util
import sys

spec = importlib.util.find_spec("lerobot.policies.molmoact2.modeling_molmoact2")
if spec is None or not spec.origin:
    print("ERROR: lerobot molmoact2 modeling module not found", file=sys.stderr)
    sys.exit(2)
PATH = spec.origin
print("patching:", PATH)
src = open(PATH).read()

if "AMD-PATCH: env dtype" in src:
    print("already patched")
    sys.exit(0)

OLD = """            trust_remote_code=trust_remote_code,
            dtype=torch.float32,
            low_cpu_mem_usage=True,"""

NEW = """            trust_remote_code=trust_remote_code,
            dtype=getattr(torch, os.environ.get("MOLMOACT2_DTYPE", "bfloat16")),  # AMD-PATCH: env dtype
            low_cpu_mem_usage=True,"""

if OLD not in src:
    print("ERROR: anchor not found; modeling file layout changed", file=sys.stderr)
    sys.exit(2)

open(PATH, "w").write(src.replace(OLD, NEW, 1))
print("patched OK -> dtype from MOLMOACT2_DTYPE (default bfloat16)")
