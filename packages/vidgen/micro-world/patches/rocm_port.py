#!/usr/bin/env python3
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Minimal, idempotent ROCm/gfx1151 source patch for AMD-AGI/Micro-World (rule 2.1: rely on
# upstream, patch only what ROCm needs). String-replacement based so it survives small upstream
# line drift. Run against a checkout: `python rocm_port.py /repos/micro-world`.
#
# What it changes and why (exactly ONE change — the rest of the stack is ROCm-clean):
#   microworld/models/wan_transformer3d.py — flash_attention() SDPA fallback.
#     flash-attn has no gfx1151 wheel, so both `import flash_attn`/`flash_attn_interface` fail and
#     FLASH_ATTN_2_AVAILABLE == FLASH_ATTN_3_AVAILABLE == False. The DiT's own attention() dispatcher
#     already handles this — its final `else` branch routes to torch.nn.functional.scaled_dot_product_
#     attention. BUT the CLIP image encoder used by the I2V / I2W paths (wan_image_encoder.py
#     AttentionPool.forward) calls flash_attention() DIRECTLY, and flash_attention()'s only non-FA3
#     path is `assert FLASH_ATTN_2_AVAILABLE` -> hard AssertionError on gfx1151. We insert an SDPA
#     fallback near the top of flash_attention() (right after the `# params` line, before the varlen
#     preprocessing) so it works without any flash-attn wheel. It is only reached with uniform-length
#     attention (q_lens/k_lens is None, the
#     AttentionPool case), where the varlen-flattened [B*L, N, C] tensors reshape cleanly back to
#     [B, N, L, C]; torch SDPA is numerically equivalent to the flash kernel (same default softmax
#     scale). When a flash-attn wheel IS present this branch is skipped entirely (upstream behavior).
import os
import sys

# Insert an SDPA fallback right after the `# params` line, i.e. BEFORE the varlen preprocessing that
# rewrites q_lens/k_lens from None to uniform tensors and flattens q/k/v. This way the fallback sees
# the original [B, L, N, C] tensors and the original (None) q_lens/k_lens. Anchored on the params
# line to tolerate upstream drift.
ATTN_OLD = (
    "    # params\n"
    "    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype\n"
)
ATTN_NEW = (
    "    # params\n"
    "    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype\n"
    "\n"
    "    # ROCm/gfx1151 port (rule 2.1): no flash-attn wheel -> fall back to torch SDPA on the\n"
    "    # original [B, L, N, C] tensors (before the varlen flatten below). Reached only via the CLIP\n"
    "    # image encoder's AttentionPool (uniform length, q_lens/k_lens=None); the DiT dispatcher\n"
    "    # SDPA-falls-back on its own. SDPA is numerically equivalent (same default softmax scale).\n"
    "    if not (FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE):\n"
    "        assert q_lens is None and k_lens is None, \\\n"
    "            'ROCm SDPA fallback supports only uniform-length attention (got q_lens/k_lens)'\n"
    "        qh = q if q.dtype in half_dtypes else q.to(dtype)\n"
    "        kh = k if k.dtype in half_dtypes else k.to(dtype)\n"
    "        vh = v if v.dtype in half_dtypes else v.to(dtype)\n"
    "        if q_scale is not None:\n"
    "            qh = qh * q_scale\n"
    "        o = torch.nn.functional.scaled_dot_product_attention(\n"
    "            qh.transpose(1, 2), kh.transpose(1, 2), vh.transpose(1, 2),\n"
    "            dropout_p=dropout_p, is_causal=causal, scale=softmax_scale)\n"
    "        return o.transpose(1, 2).type(out_dtype)\n"
)


def replace_in_file(path: str, old: str, new: str, tag: str) -> bool:
    if not os.path.isfile(path):
        print(f"  [WARN] missing: {path}")
        return False
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    if new in src:
        print(f"  [skip] already patched ({tag}): {path}")
        return True
    if old not in src:
        print(f"  [WARN] target not found ({tag}, upstream drift?): {path}")
        return False
    src = src.replace(old, new, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    print(f"  [ok] patched {tag}: {path}")
    return True


def main() -> int:
    root = sys.argv[1] if len(sys.argv) > 1 else "/repos/micro-world"
    ok = replace_in_file(
        os.path.join(root, "microworld/models/wan_transformer3d.py"),
        ATTN_OLD, ATTN_NEW, "flash_attention-SDPA-fallback",
    )
    print("rocm_port.py: done")
    # Fatal if the one required patch did not apply: the I2V/I2W CLIP path would AssertionError.
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
