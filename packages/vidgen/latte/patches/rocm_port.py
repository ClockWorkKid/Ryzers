#!/usr/bin/env python3
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Minimal, idempotent ROCm/gfx1151 source patches for Vchitect/Latte (rule 2.1: rely on
# upstream, patch only what ROCm needs). String-replacement based so it survives small
# upstream line drift. Run against a Latte checkout: `python rocm_port.py /repos/latte`.
#
# What it changes and why:
#  1. models/latte.py, models/latte_img.py — the `flash` attention branch has TWO problems:
#     (a) it wraps torch SDPA in `torch.backends.cuda.sdp_kernel(enable_math=False)`; on gfx1151
#         forcing enable_math=False (forbidding the math fallback) makes SDPA raise when no fused
#         ROCm kernel is picked; and
#     (b) it reshapes the SDPA output `(B, heads, N, head_dim)` straight to `(B, N, C)` WITHOUT the
#         `.transpose(1, 2)` that the `math` branch applies, so heads and sequence get interleaved
#         wrong. Verified on-device: as-is `flash` vs `math` cos-sim=0.38 (WRONG); adding the
#         transpose gives cos-sim=1.000000, max|Δ|=1e-7 (numerically identical to `math`).
#     Fix both: drop the context manager AND add `.transpose(1, 2)`, so `flash` is an exact,
#     ~1.8x-faster drop-in for `math` on gfx1151. `math` mode itself is untouched (default).
#     The xformers import is already guarded (try/except) and unused on ROCm, so no change needed.
#  2. tools/utils/dataset.py — the video->image dataset helper references the StyleGAN-V module
#     layout `training.dataset.ImageFolderDataset`, which does not exist in Latte's `tools/`
#     tree. Point it at the in-tree `utils.dataset.ImageFolderDataset` so the FID/IS metrics
#     (which build an image dataset from the video frame folders) resolve on any layout.
#  3. tools/metrics/metric_main.py — register `fvd_16f`: a 16-frame FVD with real-side
#     subsample_factor=1 so equal-length 16-frame clips are compared frame-for-frame (the
#     upstream `fvd2048_16f` default of 3 discards videos shorter than ~46 frames).
#  4. tools/metrics/inception_score.py — the IS split loop slices `gen_probs` using the static
#     `num_gen` (50000), so when fewer samples are available every split after the first is
#     empty and IS collapses to NaN. Split over the actual number of returned probabilities.
import sys, os

# (1) flash-SDPA on ROCm ------------------------------------------------------
SDPA_OLD = (
    "            with torch.backends.cuda.sdp_kernel(enable_math=False):\n"
    "                x = torch.nn.functional.scaled_dot_product_attention(q, k, v).reshape(B, N, C) # require pytorch 2.0"
)
SDPA_NEW = (
    "            # ROCm/gfx1151: drop forced enable_math=False sdp_kernel (raises when no fused\n"
    "            # kernel is chosen); call torch SDPA directly (fused aotriton path + math fallback).\n"
    "            # Add the .transpose(1, 2) the upstream flash branch omitted so the (B,heads,N,hd)\n"
    "            # SDPA output matches the math branch layout (verified: cos-sim 1.0 vs math).\n"
    "            x = torch.nn.functional.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, C)"
)

# (2) stale metrics dataset module name --------------------------------------
DS_OLD = "        class_name='training.dataset.ImageFolderDataset',"
DS_NEW = "        class_name='utils.dataset.ImageFolderDataset',  # ROCm port: in-tree layout"

# (4) IS split over actual sample count --------------------------------------
IS_OLD = "    scores = []\n    for i in range(num_splits):"
IS_NEW = (
    "    num_gen = len(gen_probs)  # ROCm port: split over actual sample count (supports N < 50k)\n"
    "    scores = []\n    for i in range(num_splits):"
)

# (3) 16-frame FVD registration ----------------------------------------------
FVD_ANCHOR = "@register_metric\ndef fvd2048_16f(opts):"
FVD_BLOCK = (
    "@register_metric\n"
    "def fvd_16f(opts):\n"
    "    # ROCm port: 16-frame FVD, real-side subsample_factor=1 (equal-length clip comparison).\n"
    "    fvd = frechet_video_distance.compute_fvd(\n"
    "        opts, max_real=2048, num_gen=2048, num_frames=16, realdata_subsample_factor=1)\n"
    "    return dict(fvd_16f=fvd)\n\n\n"
)


def replace_in_file(path: str, old: str, new: str, tag: str) -> bool:
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
    src = src.replace(old, new)
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    print(f"  [ok] patched {tag}: {path}")
    return True


def register_fvd16(path: str) -> bool:
    if not os.path.isfile(path):
        print(f"  [WARN] missing: {path}")
        return False
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    if "def fvd_16f(" in src:
        print(f"  [skip] fvd_16f already registered: {path}")
        return True
    if FVD_ANCHOR not in src:
        print(f"  [WARN] fvd2048_16f anchor not found (upstream drift?): {path}")
        return False
    src = src.replace(FVD_ANCHOR, FVD_BLOCK + FVD_ANCHOR, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    print(f"  [ok] registered fvd_16f: {path}")
    return True


def main() -> int:
    root = sys.argv[1] if len(sys.argv) > 1 else "/repos/latte"
    for rel in ("models/latte.py", "models/latte_img.py"):
        replace_in_file(os.path.join(root, rel), SDPA_OLD, SDPA_NEW, "flash-SDPA")
    replace_in_file(os.path.join(root, "tools/utils/dataset.py"), DS_OLD, DS_NEW, "metrics-dataset")
    replace_in_file(os.path.join(root, "tools/metrics/inception_score.py"), IS_OLD, IS_NEW, "is-split")
    register_fvd16(os.path.join(root, "tools/metrics/metric_main.py"))
    print("rocm_port.py: done")
    return 0  # non-fatal: default 'math' mode works without the patch


if __name__ == "__main__":
    raise SystemExit(main())
