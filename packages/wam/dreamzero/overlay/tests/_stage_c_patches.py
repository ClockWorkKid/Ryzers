# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Stage C overlay patch -- cache ``(clip_feas, ys, anchor_latent)``
across mid-episode KV-cache resets so K > 8 doesn't pay the 33-frame
VAE encode at every reset boundary.

Why
===
The DreamZero streaming-causal API increments
``self.current_start_frame`` by ``num_frame_per_block`` on every chunk
and triggers an internal reset
(``current_start_frame >= self.model.local_attn_size``) once the
rolling attention window fills up. After the reset the inference
re-enters the ``if current_start_frame == 0:`` branch, which calls
``self.encode_image(image, self.num_frames, ...)`` -- a 33-frame zero-
padded VAE encode of the current chunk's anchor (~10 s on Strix Halo).

For Stage B's ``max_chunk_size=4`` setting,
``local_attn_size = 4 * num_frame_per_block + 1 = 9`` (for
``num_frame_per_block=2``). So:

| K  | reset count expected           | est. wall cost of resets |
|----|--------------------------------|---------------------------|
| 4  | 0                              | 0 s                       |
| 8  | 1 (around chunk 5)             | ~10 s                     |
| 12 | 2 (around chunks 5, 9)         | ~20 s                     |
| 16 | 3 (around chunks 5, 9, 13)     | ~30 s                     |
| 20 | 4                              | ~40 s                     |

Stage C makes those reset chunks ~free by **caching the chunk-0
output of ``encode_image`` per episode** and serving the cached
``(clip_feas, ys, anchor_latent)`` on every subsequent cache-reset
within the same episode. Between episodes the cache is invalidated by
our existing between-episode VRAM purge (Stage B fix) which sets
``self.clip_feas = self.ys = None`` and our additional ``del``
of ``self._stage_c_cache``.

Semantic note
=============
The upstream behavior at a mid-episode reset is to RE-CONDITION on the
*current* chunk's anchor frame (``videos[:, :, -1:]``). Stage C
intentionally preserves the *original chunk-0 anchor* features
instead. For action prediction this is a small regression in
conditioning recency (~5 chunks worth of robot motion); the KV cache
itself still propagates the full local history, so action chunks remain
locally-coherent. For predicted-video rollouts the conditioning anchor
also stays pinned to chunk 0 across resets -- this matches the
autonomous-run patch behavior referenced in
``docs/STAGE_B_KCHUNK.md``'s K>=16 follow-up bullet.

How to use
==========
1. ``patch_kv_cache_inplace`` (Stage B patch (1)) and
   ``patch_adaptive_vae_tiles`` (Stage B (3)+(4)) must be applied
   first (Stage C does not depend on (2) ``max_chunk_size`` per se, but
   the reset boundary semantics assume the Stage B KV cache geometry).
2. Call ``apply_stage_c_patch(policy)`` after Stage B post-load
   patches. Idempotent.
3. In the per-episode loop's between-episode VRAM purge, add
   ``ah._stage_c_cache = None`` alongside the existing ``language`` /
   ``kv_cache1`` / ``kv_cache_neg`` / ``clip_feas`` / ``ys`` resets.

Disable
=======
- Env ``STAGE_C=0`` skips ``apply_stage_c_patch`` at the call site
  (the wrapper itself is gated on the cache attr, never on env).
"""
from __future__ import annotations

import os
import types
from typing import Any


def _maybe_shape(t: Any) -> str:
    """Compact ``tuple(t.shape)`` for logging; falls back gracefully."""
    try:
        return repr(tuple(t.shape))
    except Exception:
        return type(t).__name__


def patch_encode_image_cache(policy) -> bool:
    """Wrap ``WANPolicyHead.encode_image`` so that within a single
    episode, the chunk-0 output is cached on ``self_ah._stage_c_cache``
    and every subsequent call (i.e. every mid-episode cache-reset
    chunk) returns the cached triple instead of running a fresh
    33-frame VAE encode.

    The cache is keyed implicitly by "current episode": the caller is
    responsible for invalidating ``self_ah._stage_c_cache = None`` in
    the between-episode purge.

    Idempotent; re-applying on an already-patched policy is a no-op.
    """
    ah = policy.trained_model.action_head
    AhCls = type(ah)

    if getattr(AhCls, "_amd_stage_c_encode_image_patched", False):
        print("[stage-c] encode_image cache already installed")
        return False

    AhCls._amd_orig_encode_image = AhCls.encode_image

    _orig_unbound = AhCls._amd_orig_encode_image

    def _stage_c_cached_encode_image(self_ah, image, num_frames, height, width):
        cached = getattr(self_ah, "_stage_c_cache", None)
        if cached is not None:
            clip_feas, ys, new_image = cached
            self_ah._stage_c_cache_hits = (
                getattr(self_ah, "_stage_c_cache_hits", 0) + 1
            )
            print(
                f"[stage-c] reusing chunk-0 anchor features "
                f"(hit #{self_ah._stage_c_cache_hits}, clip_feas="
                f"{_maybe_shape(clip_feas)}, ys={_maybe_shape(ys)}, "
                f"new_image={_maybe_shape(new_image)}) -- skipping "
                f"{num_frames}-frame VAE encode"
            )
            return clip_feas, ys, new_image

        clip_feas, ys, new_image = _orig_unbound(
            self_ah, image, num_frames, height, width
        )
        self_ah._stage_c_cache = (clip_feas, ys, new_image)
        self_ah._stage_c_cache_misses = (
            getattr(self_ah, "_stage_c_cache_misses", 0) + 1
        )
        print(
            f"[stage-c] computed fresh anchor features "
            f"(miss #{self_ah._stage_c_cache_misses}, clip_feas="
            f"{_maybe_shape(clip_feas)}, ys={_maybe_shape(ys)}, "
            f"new_image={_maybe_shape(new_image)}) -- cached for "
            f"this episode"
        )
        return clip_feas, ys, new_image

    AhCls.encode_image = _stage_c_cached_encode_image
    AhCls._amd_stage_c_encode_image_patched = True

    print(
        "[stage-c] patch: WANPolicyHead.encode_image -> chunk-0 cache "
        "(reused on every mid-episode cache-reset; invalidated by "
        "between-episode purge via ah._stage_c_cache = None)"
    )
    return True


def invalidate_stage_c_cache(action_head) -> bool:
    """Clear the Stage C ``_stage_c_cache`` on an action_head.

    Call this from the between-episode VRAM purge in any test script
    that uses Stage C. Returns True if a cached entry was actually
    dropped.
    """
    cached = getattr(action_head, "_stage_c_cache", None)
    if cached is None:
        return False
    try:
        action_head._stage_c_cache = None
    except Exception as exc:
        print(f"[stage-c] cache invalidation failed: {exc!r}")
        return False
    return True


def apply_stage_c_patch(policy) -> None:
    """Apply the Stage C encode_image cache patch.

    Call AFTER ``apply_stage_b_post_load_patches`` and any AMD encode
    wrappers (we sit on top of those).
    """
    print("=" * 70)
    print("Stage C :: applying encode_image cache patch")
    print("=" * 70)
    patch_encode_image_cache(policy)
