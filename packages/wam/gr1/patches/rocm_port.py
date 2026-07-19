#!/usr/bin/env python3
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Minimal, idempotent ROCm/gfx1151 source patches for ByteDance/GR-1 (rule 2.1: rely on
# upstream, patch only what ROCm / portability needs). String-replacement based so it survives
# small upstream line drift. Run against a GR-1 checkout: `python rocm_port.py /repos/gr1`.
#
# What it changes and why:
#  1. models/gr1.py — the three query attention masks in forward() are created with
#     `torch.zeros((...), dtype=torch.long).cuda()`. `.cuda()` maps to HIP on ROCm and works,
#     but it hardcodes device 0 and forbids running the graph on any other explicit device
#     (e.g. a CPU parity check, or a specific `cuda:N` in multi-GPU/optimization runs). `rgb`
#     is the first forward arg and always lives on the target device, so make the masks
#     device-agnostic with `device=rgb.device`. A single replace covers all three occurrences
#     (the fwd_pred / fwd_pred_hand lines are identical). Numerically a no-op.
#  2. evaluation/calvin_evaluation.py — the MAE checkpoint is loaded with map_location='cpu'
#     but the GR-1 policy checkpoint is loaded via a bare `torch.load(policy_ckpt)`. A CUDA-
#     saved checkpoint then deserializes onto the saved ordinal, which can mismatch the visible
#     ROCm device. Add map_location='cpu' (the model is moved to `self.device` right after).
#  3. models/trajectory_gpt2.py — the vendored GPT-2 (HF ~4.5 era) has two transformers-API
#     drifts against the base image's transformers 4.36.2:
#       (a) GPT2Model is decorated with `@add_code_sample_docstrings(tokenizer_class=...)`;
#           4.x renamed that kwarg to `processor_class` -> TypeError at class definition.
#       (b) it builds attention blocks with `Block(config.n_ctx, ...)`; GPT2Config dropped
#           `n_ctx` in favour of `n_positions` (same value, 1024) -> AttributeError at init.
#     Both are mechanical renames (docs-only / config alias), numerically irrelevant.
import sys, os

# (1) device-agnostic query masks in the GR-1 forward -------------------------
CUDA_OLD = ", dtype=torch.long).cuda()"
CUDA_NEW = ", dtype=torch.long, device=rgb.device)"

# (2) robust policy checkpoint load ------------------------------------------
LOAD_OLD = "        payload = torch.load(policy_ckpt)"
LOAD_NEW = "        payload = torch.load(policy_ckpt, map_location='cpu')  # ROCm port: load off the saved ordinal"

# (3) transformers API drift: add_code_sample_docstrings dropped `tokenizer_class` -----------
DOC_OLD = "        tokenizer_class=_TOKENIZER_FOR_DOC,"
DOC_NEW = "        processor_class=_TOKENIZER_FOR_DOC,  # ROCm port: transformers>=4.x renamed tokenizer_class"

# (4) transformers API drift: GPT2Config.n_ctx removed (n_positions is the survivor) ---------
NCTX_OLD = "Block(config.n_ctx, config, scale=True)"
NCTX_NEW = "Block(config.n_positions, config, scale=True)"  # n_ctx removed from GPT2Config; n_positions is the alias


def replace_in_file(path: str, old: str, new: str, tag: str, count: int = 0) -> bool:
    if not os.path.isfile(path):
        print(f"  [WARN] missing: {path}")
        return False
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    if new in src and old not in src:
        print(f"  [skip] already patched ({tag}): {path}")
        return True
    if old not in src:
        print(f"  [WARN] target not found ({tag}, upstream drift?): {path}")
        return False
    src = src.replace(old, new) if count == 0 else src.replace(old, new, count)
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    print(f"  [ok] patched {tag}: {path}")
    return True


def main() -> int:
    root = sys.argv[1] if len(sys.argv) > 1 else "/repos/gr1"
    replace_in_file(os.path.join(root, "models/gr1.py"), CUDA_OLD, CUDA_NEW, "device-agnostic-masks")
    replace_in_file(os.path.join(root, "evaluation/calvin_evaluation.py"), LOAD_OLD, LOAD_NEW, "ckpt-map-location")
    replace_in_file(os.path.join(root, "models/trajectory_gpt2.py"), DOC_OLD, DOC_NEW, "gpt2-docstring-kwarg")
    replace_in_file(os.path.join(root, "models/trajectory_gpt2.py"), NCTX_OLD, NCTX_NEW, "gpt2-n_ctx")
    print("rocm_port.py: done")
    return 0  # non-fatal: the forward runs on ROCm even unpatched (.cuda() -> HIP)


if __name__ == "__main__":
    raise SystemExit(main())
