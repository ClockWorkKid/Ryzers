# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""ROCm compatibility patches for the upstream VLA-JEPA (starVLA) source.

Applied at build time (see Dockerfile). Kept minimal and idempotent so we stay as
close to upstream as possible (workspace rule 2.1: patch only for ROCm bugs).

Patch 1 - attention backend:
  Upstream hardcodes ``attn_implementation="flash_attention_2"`` in the Qwen VLM
  interfaces. flash-attn ships CUDA-only wheels, so on ROCm/gfx1151 we switch to
  ``"sdpa"`` (PyTorch scaled_dot_product_attention, aotriton-backed). If a ROCm
  flash-attn is installed later, set VLAJEPA_ATTN=flash_attention_2 to keep it.
"""
import os
import re
import sys

REPO = os.environ.get("VLAJEPA_REPO", "/repos/VLA-JEPA")
ATTN = os.environ.get("VLAJEPA_ATTN", "sdpa")

TARGETS = [
    "starVLA/model/modules/vlm/QWen3.py",
    "starVLA/model/modules/vlm/QWen2_5.py",
    "starVLA/model/modules/vlm/tools/add_qwen_special_tokens/add_special_tokens_to_qwen.py",
]

PATTERN = re.compile(r'attn_implementation\s*=\s*"flash_attention_2"')


def patch_file(path: str) -> bool:
    if not os.path.isfile(path):
        print(f"skip (missing): {path}")
        return False
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    new, n = PATTERN.subn(f'attn_implementation="{ATTN}"', src)
    if n == 0:
        print(f"no-op (no flash_attention_2 literal): {path}")
        return False
    with open(path, "w", encoding="utf-8") as f:
        f.write(new)
    print(f"patched {n} site(s) -> attn_implementation=\"{ATTN}\": {path}")
    return True


def main() -> int:
    if ATTN == "flash_attention_2":
        print("VLAJEPA_ATTN=flash_attention_2 -> leaving upstream attention untouched")
        return 0
    any_patched = False
    for rel in TARGETS:
        any_patched |= patch_file(os.path.join(REPO, rel))
    if not any_patched:
        print("WARNING: no attention sites patched (upstream may have changed).",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
