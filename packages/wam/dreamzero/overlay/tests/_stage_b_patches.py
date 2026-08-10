# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Stage B overlay-only ports of the four "autonomous-run" patches.

These patches restore enough VRAM headroom on the 47 GiB Strix Halo iGPU
partition to run K>=4 streaming-causal chunks of the DreamZero-DROID 14B
model in a single inference session (i.e. extend the per-episode rollout
from the Stage A K=2 cap to K=4 / K=8 / K=16).

All four upstream patches were originally `sed`-style edits to a forked
copy of `/opt/dreamzero`. We re-implement them as overlay monkey-patches
so the dreamzero source tree stays pristine.

Patch summary
=============
(1) KV-cache in-place + skip-accumulate
    - Replace ``CausalWanModel._forward_blocks`` with a faithful port that
      gates the per-block ``updated_kv_caches.append(...)`` and adds an
      in-place ``kv_cache[block_index] = updated_kv_cache`` write.
    - Replace ``WANPolicyHead._run_diffusion_steps`` so it sets the two
      DiT flags before ``self.model(...)`` and skips the redundant
      ``kv_cache[i] = updated_kv_cache.clone()`` loop.
    - Net savings during a chunk-N+1 DiT forward: ~13 GiB (no 40-tensor
      accumulation list, no 40-tensor clone).

(2) ``max_chunk_size`` 4 -> 2
    - The DZD checkpoint's ``config.json`` ships with
      ``diffusion_model_cfg.max_chunk_size = 4``. CausalWanModel uses it
      to compute ``local_attn_size = max_chunk_size * num_frame_per_block
      + 1`` which caps the streaming KV cache at 9 latent frames.
    - At max_chunk_size=2 the cap is 5 latent frames, dropping the KV cap
      from ~6.5 GiB to ~3.6 GiB (~2.9 GiB saved).
    - This patch mutates the in-memory ``cfg_dict`` BEFORE the VLAConfig
      is built, so it must run inside the low-mem ``from_pretrained``.

(3) Adaptive VAE tile geometry
    - Single-frame chunk 0 encodes a (1, 3, 480, 640) input which fits
      comfortably under (34, 34) / (18, 16) tiles.
    - 4-frame chunks (chunk N>=1) encode (4, 3, 480, 640) which inflates
      the conv3d activation 4x and OOMs at (34, 34) / (18, 16).
    - We branch tile geometry on ``videos.shape[2]`` (frames axis): use
      (14, 14) / (8, 8) when frames > 1, else keep the larger tiles.

(4) VAE on/off cycle around each per-chunk encode
    - On a tight VRAM budget, the VAE (~1 GiB resident on cuda) is pure
      overhead during the long DiT diffusion loop inside the same chunk.
    - We cycle the VAE cpu<->cuda around each ``vae.encode`` call so the
      bytes are reclaimable during the DiT forward.
    - DISABLED BY DEFAULT for Path A (no decode) and the long-video
      renderer when a subsequent ``vae.decode`` call needs VAE to be on
      cuda. Enable via env ``STAGE_B_VAE_CYCLE=1`` if you want to chase
      every last MiB.

All four patches are idempotent: re-applying them on a policy that has
already been patched is a no-op.
"""
from __future__ import annotations

import gc
import os
import types
from typing import Any


# ---------------------------------------------------------------------------
# (2) max_chunk_size config-dict override
# ---------------------------------------------------------------------------
def patch_max_chunk_size_in_cfg(cfg_dict: dict, value: int = 2) -> bool:
    """Mutate ``cfg_dict`` so the DiT is constructed with ``max_chunk_size=value``.

    Returns True if a value was changed, False if no path matched (config
    schema may have shifted).
    """
    try:
        diff_cfg = cfg_dict["action_head_cfg"]["config"]["diffusion_model_cfg"]
    except (KeyError, TypeError):
        return False
    if not isinstance(diff_cfg, dict):
        return False
    old = diff_cfg.get("max_chunk_size")
    if old == value:
        return False
    diff_cfg["max_chunk_size"] = int(value)
    print(f"[stage-b] patch (2): diffusion_model_cfg.max_chunk_size {old} -> {value}")
    return True


# ---------------------------------------------------------------------------
# (1a) Replacement _forward_blocks  --  CausalWanModel
# ---------------------------------------------------------------------------
def _amd_patched_forward_blocks(
    self,
    x,
    seq_len,
    freqs,
    timestep,
    context,
    clip_feature,
    embodiment_id,
    action,
    timestep_action,
    state,
    kv_cache,
    current_start_frame,
):
    """Faithful port of CausalWanModel._forward_blocks with two changes:

    - When ``self._amd_inplace_kv_update`` is True, ``kv_cache[i]`` is
      overwritten with the just-computed ``updated_kv_cache`` so the
      stale tensor can be freed at the next allocator pass.
    - When ``self._amd_skip_kv_accumulate`` is True, the per-block
      tensors are NOT appended to ``updated_kv_caches`` (saves ~6.5 GiB
      peak during the 40-block DiT forward).

    The public return shape is preserved (an empty list is still returned).
    """
    import torch
    from groot.vla.model.dreamzero.modules.wan2_1_submodule import (
        sinusoidal_embedding_1d,
    )

    _amd_inplace = getattr(self, "_amd_inplace_kv_update", False)
    _amd_skip_acc = getattr(self, "_amd_skip_kv_accumulate", False)

    x = x.flatten(start_dim=2).transpose(1, 2)

    B = x.shape[0]
    F = timestep.shape[1]

    if action is not None:
        embodiment_id = torch.tensor([0], device=x.device).repeat(x.shape[0])
        action_features = self.action_encoder(action, timestep_action, embodiment_id)
        state_features = self.state_encoder(state, embodiment_id)
        action_register = torch.cat([action_features, state_features], dim=1)
        action_length = action_features.shape[1]
        action_register_length = action_register.shape[1]
        x = torch.cat([x, action_register], dim=1)
    else:
        action_features = None
        state_features = None
        action_length = 0
        action_register_length = None

    if F <= seq_len:
        repeat = (seq_len + F - 1) // F
        timestep = timestep.repeat_interleave(repeat, dim=1)[:, :seq_len]
    else:
        indices = torch.linspace(0, F - 1, seq_len, device=timestep.device, dtype=torch.long)
        timestep = timestep[:, indices]

    if action is not None:
        assert timestep_action is not None
        assert state_features is not None
        stride = timestep_action.shape[1] // state_features.shape[1]
        timestep_state = timestep_action[:, ::stride]
        timestep = torch.cat([timestep, timestep_action, timestep_state], dim=1)

    e = self.time_embedding(
        sinusoidal_embedding_1d(self.freq_dim, timestep.flatten()).type_as(x)
    )
    e = e.unflatten(dim=0, sizes=(B, -1))
    e0 = self.time_projection(e)
    e0 = e0.unflatten(dim=2, sizes=(6, self.dim))

    context = self.text_embedding(context)
    if clip_feature is not None:
        clip_embedding = self.img_emb(clip_feature)
        context = torch.cat([clip_embedding, context], dim=1)

    updated_kv_caches: list[Any] = []
    for block_index, block in enumerate(self.blocks):
        x, updated_kv_cache = block(
            x=x,
            e=e0,
            freqs=freqs,
            freqs_action=self.freqs_action,
            freqs_state=self.freqs_state,
            context=context,
            action_register_length=action_register_length,
            kv_cache=kv_cache[block_index],
            current_start_frame=current_start_frame,
        )
        if _amd_inplace:
            kv_cache[block_index] = updated_kv_cache
        if not _amd_skip_acc:
            updated_kv_caches.append(updated_kv_cache)
        else:
            del updated_kv_cache

    if action is not None:
        action_noise_pred = x[:, seq_len: seq_len + action_length]
        action_noise_pred = self.action_decoder(action_noise_pred, embodiment_id)
    else:
        action_noise_pred = None

    x_video = x[:, :seq_len]
    e_video = e[:, :seq_len]
    x_video = self.head(x_video, e_video.unsqueeze(2))

    return x_video, action_noise_pred, updated_kv_caches


# ---------------------------------------------------------------------------
# (1b) Replacement _run_diffusion_steps  --  WANPolicyHead
# ---------------------------------------------------------------------------
def _amd_patched_run_diffusion_steps(
    self_ah,
    noisy_input,
    timestep,
    action,
    timestep_action,
    state,
    embodiment_id,
    context,
    seq_len,
    y,
    clip_feature,
    kv_caches,
    crossattn_caches,
    kv_cache_metadata,
):
    """Faithful port of WANPolicyHead._run_diffusion_steps with two changes:

    - Before each ``self.model(...)`` call, set the model-level flags so
      our patched ``_forward_blocks`` performs the in-place mutation and
      skips the accumulation list (saves ~6.5 GiB peak).
    - Drop the trailing ``for ... kv_cache[i] = updated_kv_cache.clone()``
      loop because the model has already mutated ``kv_cache`` in-place.
      Removing the loop also frees the ~6.5 GiB clone allocation.

    Net VRAM saving on chunk-N+1 DiT forward: ~13 GiB.
    """
    import torch

    predictions = []
    for index, prompt_emb in enumerate(context):
        kv_cache = kv_caches[index]
        crossattn_cache = crossattn_caches[index]
        if (not kv_cache_metadata["update_kv_cache"]) and self_ah.trt_engine is not None:
            obs_noise_pred, action_noise_pred = self_ah.trt_engine(
                noisy_input,
                timestep,
                action=action,
                timestep_action=timestep_action,
                state=state,
                context=prompt_emb,
                y=y,
                clip_feature=clip_feature,
                kv_cache=kv_cache,
            )
        else:
            _amd_update = bool(kv_cache_metadata["update_kv_cache"])
            self_ah.model._amd_inplace_kv_update = _amd_update
            self_ah.model._amd_skip_kv_accumulate = True
            obs_noise_pred, action_noise_pred, _updated = self_ah.model(
                noisy_input,
                timestep,
                action=action,
                timestep_action=timestep_action,
                state=state,
                embodiment_id=embodiment_id,
                context=prompt_emb,
                seq_len=seq_len,
                y=y,
                clip_feature=clip_feature,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start_frame=kv_cache_metadata["start_frame"],
            )
            del _updated
        obs_noise_pred = obs_noise_pred.clone()
        if action_noise_pred is not None:
            action_noise_pred = action_noise_pred.clone()
        else:
            action_noise_pred = torch.tensor(0.0, device=obs_noise_pred.device)
        predictions.append((obs_noise_pred, action_noise_pred))
    return self_ah._exchange_predictions(predictions)


def patch_kv_cache_inplace(policy) -> bool:
    """Install patches (1a) and (1b). Idempotent."""
    ah = policy.trained_model.action_head
    DitCls = type(ah.model)
    AhCls = type(ah)

    if getattr(DitCls, "_amd_inplace_patched", False):
        print("[stage-b] patch (1): kv-cache in-place already installed")
        return False

    DitCls._amd_orig_forward_blocks = DitCls._forward_blocks
    DitCls._forward_blocks = _amd_patched_forward_blocks
    DitCls._amd_inplace_patched = True
    print("[stage-b] patch (1a): CausalWanModel._forward_blocks -> in-place + skip-accumulate")

    AhCls._amd_orig_run_diffusion_steps = AhCls._run_diffusion_steps
    AhCls._run_diffusion_steps = _amd_patched_run_diffusion_steps
    AhCls._amd_run_diffusion_steps_patched = True
    print("[stage-b] patch (1b): WANPolicyHead._run_diffusion_steps -> flag-set + clone-loop dropped")
    return True


# ---------------------------------------------------------------------------
# (3) Adaptive VAE tile geometry  +  (4) optional VAE cpu<->cuda cycle
# ---------------------------------------------------------------------------
def patch_adaptive_vae_tiles(
    policy,
    *,
    single_frame_tile: tuple = (34, 34),
    single_frame_stride: tuple = (18, 16),
    multi_frame_tile: tuple = (14, 14),
    multi_frame_stride: tuple = (8, 8),
    cycle_offload: bool | None = None,
) -> bool:
    """Wrap vae.encode so it:
      - always passes ``tiled=True``
      - selects tile geometry by input frame count
      - optionally cycles VAE cpu<->cuda around the encode call

    ``cycle_offload``:
      - None  : read env ``STAGE_B_VAE_CYCLE`` (truthy -> on)
      - True  : force on   (saves ~1 GiB but breaks subsequent vae.decode
                            unless caller pushes vae back to cuda)
      - False : force off

    Idempotent: replaces any existing wrapped ``vae.encode``.
    """
    import torch

    if cycle_offload is None:
        cycle_offload = os.environ.get("STAGE_B_VAE_CYCLE", "0").strip().lower() in (
            "1", "true", "yes", "on",
        )

    vae = policy.trained_model.action_head.vae
    VaeCls = type(vae)

    if not hasattr(VaeCls, "_amd_orig_encode"):
        VaeCls._amd_orig_encode = VaeCls.encode

    _orig_unbound = VaeCls._amd_orig_encode

    def _adaptive_tiled_vae_encode(self_vae, videos, tiled=False, tile_size=None, tile_stride=None):
        # videos shape: (B, C, F, H, W) -- F is the frame axis.
        frames = videos.shape[2] if videos.ndim >= 3 else 1

        if frames > 1:
            tsize = multi_frame_tile
            tstride = multi_frame_stride
        else:
            tsize = single_frame_tile
            tstride = single_frame_stride

        if cycle_offload:
            try:
                if next(self_vae.parameters()).device.type != "cuda":
                    self_vae.to(device=videos.device, dtype=torch.bfloat16)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            except StopIteration:
                pass

        out = _orig_unbound(
            self_vae, videos, tiled=True, tile_size=tsize, tile_stride=tstride,
        )

        if cycle_offload:
            try:
                self_vae.cpu()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as exc:
                print(f"[stage-b] vae cpu offload skipped: {exc!r}")

        return out

    vae.encode = types.MethodType(_adaptive_tiled_vae_encode, vae)
    print(
        f"[stage-b] patch (3+4): vae.encode -> adaptive tiles"
        f" (1-frame={single_frame_tile}/{single_frame_stride},"
        f" multi-frame={multi_frame_tile}/{multi_frame_stride},"
        f" cycle_offload={cycle_offload})"
    )
    return True


# ---------------------------------------------------------------------------
# Convenience: apply all post-load Stage B patches.
# ---------------------------------------------------------------------------
def apply_stage_b_post_load_patches(policy, *, vae_cycle: bool | None = None) -> None:
    """Apply patches (1) and (3+4). Patch (2) must be applied earlier
    (inside _amd_low_mem_from_pretrained on the cfg_dict) before policy
    construction.
    """
    print("=" * 70)
    print("Stage B :: applying post-load overlay patches")
    print("=" * 70)
    patch_kv_cache_inplace(policy)
    patch_adaptive_vae_tiles(policy, cycle_offload=vae_cycle)
