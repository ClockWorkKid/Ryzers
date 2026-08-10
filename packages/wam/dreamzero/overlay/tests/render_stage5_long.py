#!/usr/bin/env python3
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Stage 5 LONG-FORM rollout -- multi-chunk video render with Stage B
overlay (KV-cache in-place + adaptive VAE tiles + max_chunk_size override).

For each requested episode this script:
  1. Builds the AR-style frame schedule for K chunks:
       chunk 0  : [0]
       chunk i+1: [c-23, c-16, c-8, c]   with c = 23 + i*24
  2. Calls ``policy.lazy_joint_forward_causal`` K times so the action_head
     accumulates its streaming KV cache across the K chunks.
  3. Captures each chunk's ``video_pred`` latent, VAE-decodes to RGB, and
     concatenates the per-chunk frames into a single long mp4 of length
     ~K * 2.2 s @ 15 fps (each chunk decodes to 33 frames).
  4. Also writes a side-by-side mp4 (anchor stitched on the left of every
     concatenated frame).

Reuses (verbatim where possible):
  * ``DroidDataset`` from ``validate_stage3_path_b.py``
  * The 10-patch AMD ROCm suite + perf overlay + Stage B helpers
  * VAE decode / mp4 / side-by-side helpers from ``render_stage5_gallery.py``

Inputs (read-only):
  - $MODEL_PATH                 DreamZero-DROID snapshot dir
  - $WAN21_DIR                  Wan2.1 snapshot mount
  - $DROID_DATASET_DIR          lerobot/droid_1.0.1 local dir
                                (default /data/lerobot_droid_101)
  - $EPISODES                   comma-sep list of episode_index to render
                                (default 1,13,22,82  - representative subset)
  - $NUM_CHUNKS                 K chunks per episode (default 4; try 8 once
                                K=4 has passed)
  - $OUTPUT_DIR                 where to dump artifacts
  - $VIDEO_FPS                  default 5 (model's predicted-video native rate;
                                upstream socket_test_optimized_AR.py also saves
                                at fps=5 — see docs/STAGE_H_TIME_AXIS_BUG.md).
                                Setting this to 15 (the DROID dataset rate)
                                makes predicted videos play 3.2x too fast and
                                appear to "loop / complete the task multiple
                                times" — that's a playback-rate bug, not a
                                model bug.
  - $STREAMING_OVERLAP_DECODE   default 1. Decode each per-chunk video with a
                                rolling latent overlap window so the VAE sees
                                prior temporal context at chunk boundaries.
  - $STREAMING_OVERLAP_K        default 1 latent frame of context.
  - $ANCHOR_LOCAL_FRAME         optional, default 23
  - $ENABLE_FLASH_SDPA          "1" turns on aotriton flash + mem-eff SDPA
  - $DENOISE_STEPS              optional int (Stage A default: 2)
  - $STAGE_B                    "1" applies the 4 Stage B patches (default
                                ON for this script -- it's the whole point)
  - $STAGE_B_MAX_CHUNK_SIZE     default 2
  - $STAGE_B_VAE_CYCLE          default 0 (cycle would break VAE decode)

Outputs (under $OUTPUT_DIR):
  - long_manifest.json
  - episode_NNN/
      manifest.json
      long_video.mp4              (~K*2.2 s concatenated rollout)
      long_video_vs_anchor.mp4    (anchor left | rollout right)
      action_chunks.npz           (K, 24, 8) + concat trajectory
      trajectory.png              concat 24*K-step plot
      per_chunk_video_NN.mp4      individual chunk decodes (debug)
      mem_per_chunk.csv

Exit codes:
  0  >=1 episode produced a long video
  2  base import failed
  3  model/dataset import failed
  4  policy load failed
  5  fatal failure on chunk 0 of first episode
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
import traceback
import types
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# overlay/tests on sys.path so we can import the Stage B helper module +
# share the gallery's helpers (VAE decode, mp4 writer, side-by-side).
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))


# ---------------------------------------------------------------------------
# Stage B master flags
# ---------------------------------------------------------------------------
_STAGE_B_ENABLED = os.environ.get("STAGE_B", "1").strip().lower() in (
    "1", "true", "yes", "on",
)
_STAGE_B_MAX_CHUNK_SIZE = int(os.environ.get("STAGE_B_MAX_CHUNK_SIZE", "2"))

# Stage C: cache (clip_feas, ys, anchor_latent) so K > 8 doesn't pay
# the 33-frame VAE encode at every local_attn_size boundary.
_STAGE_C_ENABLED = os.environ.get("STAGE_C", "0").strip().lower() in (
    "1", "true", "yes", "on",
)

# Debug-only continuous-video pass. When set, after the production
# observation-grounded per-chunk loop we replay the same schedule passing
# each chunk's video_pred as the next chunk's latent_video to upstream
# WANPolicyHead.lazy_joint_video_action -- this makes the model auto-
# regressively extend its own predicted latent stream (bypasses the
# per-chunk observation VAE re-encode at line 1080-1100 of
# wan_flow_matching_action_tf.py) and produces a continuous predicted-
# future video. Actions from this pass are DISCARDED; production action
# chunks come from the observation-grounded pass.
_LONG_VIDEO_AUTOREGRESSIVE_DEBUG = os.environ.get(
    "LONG_VIDEO_AUTOREGRESSIVE_DEBUG", "1"
).strip().lower() in ("1", "true", "yes", "on")


def banner(msg: str) -> None:
    bar = "=" * max(len(msg), 60)
    print(f"\n{bar}\n{msg}\n{bar}", flush=True)


def cuda_mem_gib(reset: bool = False) -> dict[str, float]:
    import torch
    if not torch.cuda.is_available():
        return {"alloc_gib": 0.0, "max_alloc_gib": 0.0, "reserved_gib": 0.0}
    torch.cuda.synchronize()
    out = {
        "alloc_gib":     torch.cuda.memory_allocated() / 1024**3,
        "max_alloc_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "reserved_gib":  torch.cuda.memory_reserved() / 1024**3,
    }
    if reset:
        torch.cuda.reset_peak_memory_stats()
    return out


def host_mem_gib() -> dict[str, float]:
    try:
        import psutil
        m = psutil.virtual_memory()
        return {"used_gib": (m.total - m.available) / 1024**3, "free_gib": m.available / 1024**3}
    except Exception:
        return {"used_gib": -1.0, "free_gib": -1.0}


# ---------------------------------------------------------------------------
# Frame schedule (identical to validate_stage3_path_a.py)
# ---------------------------------------------------------------------------
RELATIVE_OFFSETS = (-23, -16, -8, 0)
ACTION_HORIZON = 24
PRIMARY_NATIVE_CAM = "video.exterior_image_1_left"


def build_long_schedule(anchor_local: int, ep_len: int, num_chunks: int) -> list[list[int]]:
    """Multi-chunk schedule starting at ``anchor_local`` (chunk 1's "c").

    Chunk 0 is the seed single-frame [0]; subsequent chunks step forward by
    24 frames so the model rolls through the episode. We clamp to ep_len-1
    if the schedule runs off the end (matches Path A behavior).
    """
    schedule: list[list[int]] = [[0]]
    c = anchor_local
    for _ in range(1, num_chunks):
        idx = [max(min(c + off, ep_len - 1), 0) for off in RELATIVE_OFFSETS]
        if idx[-1] >= ep_len:
            break
        schedule.append(idx)
        c += ACTION_HORIZON
    return schedule


# ---------------------------------------------------------------------------
# Reuse the helpers from render_stage5_gallery.py (decode_video_pred,
# write_mp4, make_side_by_side, extract_action_chunk) WITHOUT re-importing
# the whole gallery (which would execute its main). We pull the implementations
# verbatim here so this script is self-contained.
# ---------------------------------------------------------------------------
def write_mp4(path: Path, frames_u8, fps: int) -> None:
    """frames_u8: (T, H, W, 3) uint8 RGB"""
    import av
    import numpy as np

    T, H, W, _ = frames_u8.shape
    container = av.open(str(path), mode="w")
    try:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = W
        stream.height = H
        stream.pix_fmt = "yuv420p"
        stream.options = {"preset": "fast", "crf": "20"}
        for i in range(T):
            frame = av.VideoFrame.from_ndarray(frames_u8[i], format="rgb24")
            for pkt in stream.encode(frame):
                container.mux(pkt)
        for pkt in stream.encode():
            container.mux(pkt)
    finally:
        container.close()


def make_side_by_side(anchor_rgb, frames_u8):
    """Left = anchor image (held), right = rollout frames."""
    import numpy as np
    H = max(anchor_rgb.shape[0], frames_u8.shape[1])
    W = anchor_rgb.shape[1] + frames_u8.shape[2]
    T = frames_u8.shape[0]
    canvas = np.zeros((T, H, W, 3), dtype=np.uint8)
    a = anchor_rgb
    canvas[:, :a.shape[0], :a.shape[1]] = a
    canvas[:, :frames_u8.shape[1], a.shape[1]:] = frames_u8
    return canvas, a


def decode_video_pred(policy, video_pred):
    """Run the VAE decoder to convert the (1, 16, 3, 44, 80) latent to a
    (T, H, W, 3) uint8 RGB tensor. Ensures the VAE is on cuda before the
    call (required after Stage B's optional VAE cpu offload).
    """
    import torch
    import numpy as np

    t0 = time.perf_counter()
    ah = policy.trained_model.action_head
    vae = ah.vae

    if next(vae.parameters()).device.type != "cuda":
        vae.to(device=torch.device("cuda"), dtype=torch.bfloat16)
        torch.cuda.empty_cache()

    if not torch.is_tensor(video_pred):
        raise RuntimeError(f"video_pred not a tensor: {type(video_pred)}")
    z = video_pred.to(device="cuda", dtype=torch.bfloat16)

    with torch.no_grad():
        decoded = vae.decode(
            z,
            tiled=True,
            tile_size=(34, 34),
            tile_stride=(18, 16),
        )

    # decoded: (1, 3, F, H, W) bf16 in [-1, 1]
    if decoded.ndim == 5:
        decoded = decoded[0]
    decoded = decoded.permute(1, 2, 3, 0)  # (F, H, W, 3)
    decoded = decoded.clamp(-1.0, 1.0)
    decoded = ((decoded + 1.0) * 127.5).to(torch.uint8).cpu().numpy()
    dt = time.perf_counter() - t0
    return decoded, dt


def decode_video_pred_overlap(policy, video_pred, prev_tail, k_overlap: int):
    """Streaming-compatible VAE decode with K latent frames of left context.

    ``video_pred`` is the new chunk latent. ``prev_tail`` is the last K latent
    frames from the previous chunk, CPU or CUDA. The VAE emits one RGB frame
    from the first latent and four RGB frames for each subsequent latent, so we
    skip the RGB frames generated from the overlap context and keep only the new
    chunk's contribution.
    """
    import torch

    if k_overlap <= 0 or prev_tail is None:
        frames_u8, dt = decode_video_pred(policy, video_pred)
        z_new = video_pred.detach()
        k_keep = max(1, min(1 if k_overlap <= 0 else k_overlap, int(z_new.shape[2])))
        new_tail = z_new[:, :, -k_keep:, :, :].to(
            device="cpu", dtype=torch.bfloat16
        ).contiguous()
        return frames_u8, dt, new_tail, {
            "mode": "isolated" if k_overlap <= 0 else "overlap_first_chunk",
            "k_overlap": int(k_overlap),
            "decode_input_latent_frames": int(z_new.shape[2]),
            "skipped_rgb_frames": 0,
            "emitted_rgb_frames": int(frames_u8.shape[0]),
        }

    z_new = video_pred.to(device="cuda", dtype=torch.bfloat16)
    k_keep = min(int(k_overlap), int(z_new.shape[2]))
    ctx = prev_tail[:, :, -int(k_overlap):, :, :].to(
        device="cuda", dtype=torch.bfloat16
    )
    decode_input = torch.cat([ctx, z_new], dim=2)
    n_skip = 1 + 4 * (int(ctx.shape[2]) - 1)

    frames_full, dt = decode_video_pred(policy, decode_input)
    frames_u8 = frames_full[n_skip:]
    new_tail = z_new[:, :, -k_keep:, :, :].detach().to(
        device="cpu", dtype=torch.bfloat16
    ).contiguous()
    return frames_u8, dt, new_tail, {
        "mode": "overlap",
        "k_overlap": int(k_overlap),
        "decode_input_latent_frames": int(decode_input.shape[2]),
        "skipped_rgb_frames": int(n_skip),
        "emitted_rgb_frames": int(frames_u8.shape[0]),
    }


def extract_action_chunk(batch_out):
    """Pull the (24, 8) joint+gripper from the model output Batch."""
    import numpy as np
    import torch
    from tianshou.data import Batch

    def to_np(v):
        if hasattr(v, "detach"):
            t = v.detach().cpu()
            if t.dtype in (torch.bfloat16, torch.float16):
                t = t.to(torch.float32)
            return t.numpy()
        return np.asarray(v)

    act = getattr(batch_out, "act", None)
    if act is None:
        return None
    flat: dict[str, Any] = {}
    if isinstance(act, Batch) or hasattr(act, "keys"):
        for k in act.keys():
            v = act[k]
            if isinstance(v, Batch) and hasattr(v, "keys"):
                for k2 in v.keys():
                    flat[f"{k}.{k2}"] = v[k2]
            else:
                flat[k] = v
    joint = next((to_np(v) for k, v in flat.items() if "joint_position" in k), None)
    gripper = next((to_np(v) for k, v in flat.items() if "gripper_position" in k), None)
    if joint is None:
        return None
    j = joint.astype("float32")
    while j.ndim > 2 and j.shape[0] == 1:
        j = j[0]
    if gripper is not None:
        g = gripper.astype("float32")
        while g.ndim > 2 and g.shape[0] == 1:
            g = g[0]
    else:
        g = np.zeros(j.shape[:-1] + (1,), dtype="float32")
    if g.shape[:-1] != j.shape[:-1]:
        g = np.broadcast_to(g, j.shape[:-1] + (g.shape[-1],)).copy()
    return np.concatenate([j, g], axis=-1)


# ---------------------------------------------------------------------------
# Debug-only: continuous-predicted-video pass.
#
# Re-runs the same per-chunk schedule but feeds chunk N>0's prediction back
# in as `latent_video`, the kwarg already plumbed all the way through
#   sim_policy.lazy_joint_forward_causal(batch, latent_video=...)
#   base_vla.lazy_joint_video_action_causal(inputs, latent_video=...)
#   WANPolicyHead.lazy_joint_video_action(..., latent_video=...)
# When `latent_video is not None and current_start_frame != 0` upstream
# sets `image = latent_video` (line 1077) instead of re-running
# self.vae.encode() over the observation frames. The KV-cache ref pass
# (line 1167-1190) then re-anchors on the model's OWN previous predictions
# (idempotent: they were already in cache from the prediction step), and
# the next prediction is a true continuation -- gives a smooth continuous
# predicted-future video. Actions diverge from real observations after
# chunk 0; we throw them away.
# ---------------------------------------------------------------------------
def _reset_action_head_episode_state(policy):
    import torch
    ah = policy.trained_model.action_head
    for _attr in ("language", "clip_feas", "ys", "kv_cache1", "kv_cache_neg",
                  "crossattn_cache", "crossattn_cache_neg", "image"):
        if hasattr(ah, _attr):
            try:
                setattr(ah, _attr, None)
            except Exception:
                pass
    if hasattr(ah, "current_start_frame"):
        try:
            ah.current_start_frame = 0
        except Exception:
            pass
    if hasattr(ah, "_stage_c_cache"):
        try:
            ah._stage_c_cache = None
        except Exception:
            pass
    if hasattr(policy, "_stage_b_state"):
        try:
            policy._stage_b_state["prompt_emb_cache"].clear()
        except Exception:
            pass
    import gc as _gc
    _gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _run_autoregressive_debug_pass(policy, ds, ep, ep_tag, schedule, prompt):
    """Replay the per-chunk schedule in pure-autoregressive video mode.

    Returns (unified_rgb_uint8, n_latent_frames, total_wall_seconds).
    Falls back to (None, 0, 0.0) on first-chunk failure.
    """
    import numpy as np
    import torch
    from tianshou.data import Batch

    banner(f"[{ep_tag}] autoregressive video debug pass (LONG_VIDEO_AUTOREGRESSIVE_DEBUG=1)")
    _reset_action_head_episode_state(policy)

    latent_video_carry = None
    ar_latents_cpu: list = []
    t_total = time.perf_counter()

    for c_idx, frame_indices in enumerate(schedule):
        try:
            frames_dict = ds.read_frames(ep, frame_indices)
        except Exception as exc:
            print(f"[{ep_tag} AR chunk {c_idx}] read_frames failed: {exc!r}")
            break

        obs: dict[str, Any] = dict(frames_dict)
        try:
            anchor_state_8 = ds.read_state(ep, frame_indices[-1])
            if anchor_state_8.shape == (8,):
                js = anchor_state_8[:7].reshape(1, 7).astype(np.float64)
                gs = anchor_state_8[7:8].reshape(1, 1).astype(np.float64)
            else:
                js = np.zeros((1, 7), dtype=np.float64)
                gs = np.zeros((1, 1), dtype=np.float64)
        except Exception:
            js = np.zeros((1, 7), dtype=np.float64)
            gs = np.zeros((1, 1), dtype=np.float64)
        obs["state.joint_position"] = js
        obs["state.gripper_position"] = gs
        obs["annotation.language.action_text"] = prompt

        t0 = time.perf_counter()
        try:
            batch_in = Batch(obs=obs)
            if c_idx == 0 or latent_video_carry is None:
                result = policy.lazy_joint_forward_causal(batch_in)
            else:
                lv_cuda = latent_video_carry.to(device="cuda",
                                                dtype=torch.bfloat16)
                result = policy.lazy_joint_forward_causal(
                    batch_in, latent_video=lv_cuda
                )
            torch.cuda.synchronize()
        except Exception as exc:
            print(f"[{ep_tag} AR chunk {c_idx}] inference failed: {exc!r}")
            traceback.print_exc()
            break
        dt = time.perf_counter() - t0

        if not isinstance(result, tuple) or len(result) < 2 or result[1] is None:
            print(f"[{ep_tag} AR chunk {c_idx}] no video_pred")
            break
        video_pred = result[1]
        try:
            latent_video_carry = video_pred.detach().to(
                device="cpu", dtype=torch.bfloat16).contiguous()
            ar_latents_cpu.append(latent_video_carry)
        except Exception as e:
            print(f"[{ep_tag} AR chunk {c_idx}] latent cpu snapshot failed: {e!r}")
            break

        mode = "cold/anchor" if c_idx == 0 else "latent_video=prev"
        print(f"[{ep_tag} AR chunk {c_idx}] frames={frame_indices}  "
              f"infer={dt:.1f}s  mode={mode}  "
              f"video_pred_shape={tuple(video_pred.shape)}")

        try:
            del result, video_pred
        except Exception:
            pass
        import gc as _gc
        _gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    if not ar_latents_cpu:
        return None, 0, time.perf_counter() - t_total

    try:
        latent_cat = torch.cat(ar_latents_cpu, dim=2)
        n_lat = latent_cat.shape[2]
        ar_rgb, _ = decode_video_pred(policy, latent_cat)
    except Exception as e:
        print(f"[{ep_tag} AR unified decode failed: {e!r}")
        traceback.print_exc()
        return None, 0, time.perf_counter() - t_total

    t_total = time.perf_counter() - t_total
    return ar_rgb, n_lat, t_total


# ---------------------------------------------------------------------------
# 10-patch AMD ROCm suite -- IDENTICAL set to Stage 2 v16 / gallery.
# (Verbatim from render_stage5_gallery.py)
# ---------------------------------------------------------------------------
def apply_amd_patches(policy):
    import torch
    import tree as _tree

    banner("Applying 10-patch AMD ROCm suite (same as Stage 2 v16)")
    vae = policy.trained_model.action_head.vae
    _orig_vae_encode_unbound = type(vae).encode

    def _tiled_vae_encode(self_vae, videos, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        return _orig_vae_encode_unbound(self_vae, videos, tiled=True,
                                        tile_size=tile_size, tile_stride=tile_stride)
    vae.encode = types.MethodType(_tiled_vae_encode, vae)
    print("[patch] vae.encode -> always tiled=True (will be overridden by Stage B adaptive)")

    # Stage H G1a: batched spatial tiles in tiled_decode. NEGATIVE RESULT.
    #
    # Hypothesis: the upstream WanVideoVAE.tiled_decode iterates spatial
    # tiles serially in a Python for-loop, and within each tile the
    # per-frame temporal decode also runs in a Python for-loop. For an
    # N_tile=8 spatial grid (our 44x80 latent at tile_size=(34,34),
    # stride=(18,16)) and T=320 latent frames at K=128, that's 8 * 320 =
    # 2560 sequential decoder calls. We hypothesized that Python +
    # kernel-launch overhead dominated and batching all spatial tiles
    # per latent frame into one batched self.model.decode call would
    # give a 3-5x speedup.
    #
    # Measurement (K=8 ep1 smoke):
    #   baseline (serial):    unified_decode=64.7s  VRAM peak=38.03 GiB
    #   batched (this patch): unified_decode=74.5s  VRAM peak=42.41 GiB
    #   --> +15% wall, +4.4 GiB VRAM. Per-chunk decode also ~20% slower.
    #
    # Root cause: rocm-smi shows GPU at 100% utilization in the serial
    # path -- there is no Python idle time to amortize away. The Wan
    # VAE decoder is memory-bandwidth-bound on Strix Halo's unified-
    # memory APU (not compute or launch-overhead bound), and growing
    # the per-call working set N_tile-fold thrashes the cache without
    # giving any compute parallelism win.
    #
    # The code path is preserved but disabled by default for two reasons:
    #   1. A discrete-GPU port (where kernel-launch overhead is real and
    #      memory bandwidth >> APU) would likely flip the trade-off.
    #   2. Smaller batch sizes (2-4 tiles instead of 8) might find a
    #      sweet spot worth investigating.
    #
    # See docs/STAGE_H_G1A_NEGATIVE.md for full investigation.
    # Toggle: STAGE_H_BATCHED_VAE_DECODE=1 forces the batched path on.
    stage_h_batched = os.environ.get(
        "STAGE_H_BATCHED_VAE_DECODE", "0"
    ).strip().lower() in ("1", "true", "yes", "on")
    if stage_h_batched:
        import torch.nn.functional as _F

        _orig_tiled_decode_unbound = type(vae).tiled_decode

        def _batched_tiled_decode(self_vae, hidden_states, tile_size, tile_stride):
            _, _, T, H, W = hidden_states.shape
            size_h, size_w = tile_size
            stride_h, stride_w = tile_stride
            upsample = self_vae.upsampling_factor

            tasks = []
            for h in range(0, H, stride_h):
                if (h - stride_h >= 0 and h - stride_h + size_h >= H):
                    continue
                for w in range(0, W, stride_w):
                    if (w - stride_w >= 0 and w - stride_w + size_w >= W):
                        continue
                    h_, w_ = h + size_h, w + size_w
                    tasks.append((h, h_, w, w_))

            if not tasks:
                return _orig_tiled_decode_unbound(self_vae, hidden_states, tile_size, tile_stride)

            out_T = T * 4 - 3
            weight = torch.zeros(
                (1, 1, out_T, H * upsample, W * upsample),
                dtype=hidden_states.dtype, device=hidden_states.device,
            )
            values = torch.zeros(
                (1, 3, out_T, H * upsample, W * upsample),
                dtype=hidden_states.dtype, device=hidden_states.device,
            )

            # Slice + pad each tile to (size_h, size_w); track owned size.
            tile_inputs = []
            tile_owned_hw = []
            for h, h_, w, w_ in tasks:
                h_real = min(h_, H) - h
                w_real = min(w_, W) - w
                tile = hidden_states[:, :, :, h:h + h_real, w:w + w_real]
                pad_h = size_h - h_real
                pad_w = size_w - w_real
                if pad_h > 0 or pad_w > 0:
                    # 5D tensor (B, C, T, H, W): F.pad in non-constant
                    # mode requires length-6 pad (W_left, W_right, H_top,
                    # H_bottom, T_left, T_right). Pad only on H/W bottom/right
                    # since edge tiles always overrun the high index side.
                    tile = _F.pad(tile, (0, pad_w, 0, pad_h, 0, 0), mode="replicate")
                tile_inputs.append(tile)
                tile_owned_hw.append((h_real, w_real))

            batched = torch.cat(tile_inputs, dim=0)

            with torch.no_grad():
                batched_out = self_vae.model.decode(batched, self_vae.scale)

            for i, (h, h_, w, w_) in enumerate(tasks):
                h_real, w_real = tile_owned_hw[i]
                decoded_h = h_real * upsample
                decoded_w = w_real * upsample
                tile_decoded = batched_out[i:i + 1, :, :, :decoded_h, :decoded_w]
                mask = self_vae.build_mask(
                    tile_decoded,
                    is_bound=(h == 0, h_ >= H, w == 0, w_ >= W),
                    border_width=(
                        (size_h - stride_h) * upsample,
                        (size_w - stride_w) * upsample,
                    ),
                ).to(dtype=hidden_states.dtype)
                target_h = h * upsample
                target_w = w * upsample
                values[:, :, :, target_h:target_h + tile_decoded.shape[3],
                       target_w:target_w + tile_decoded.shape[4]] += tile_decoded * mask
                weight[:, :, :, target_h:target_h + tile_decoded.shape[3],
                       target_w:target_w + tile_decoded.shape[4]] += mask

            values = values / weight
            values = values.clamp_(-1, 1)
            return values

        vae.tiled_decode = types.MethodType(_batched_tiled_decode, vae)
        print(f"[patch] STAGE_H_BATCHED_VAE_DECODE=1: vae.tiled_decode -> batched "
              f"(n_spatial_tiles per decode call, expect ~3-5x decode speedup)")
    else:
        print("[patch] STAGE_H_BATCHED_VAE_DECODE=0: keeping upstream serial tiled_decode")

    _orig_lazy_causal = type(policy.trained_model).lazy_joint_video_action_causal

    def _cuda_hoisted_lazy_causal(self_vla, inputs, latent_video=None):
        moved = 0
        if isinstance(inputs, dict):
            for k, v in list(inputs.items()):
                if torch.is_tensor(v) and v.device.type == "cpu":
                    inputs[k] = v.to("cuda", non_blocking=True)
                    moved += 1
        if latent_video is not None and torch.is_tensor(latent_video) and latent_video.device.type == "cpu":
            latent_video = latent_video.to("cuda", non_blocking=True)
            moved += 1
        return _orig_lazy_causal(self_vla, inputs, latent_video=latent_video)
    type(policy.trained_model).lazy_joint_video_action_causal = _cuda_hoisted_lazy_causal

    def _cuda_prepare_input(self_vla, inputs):
        self_vla.validate_inputs(inputs)
        backbone_inputs = self_vla.backbone.prepare_input(inputs)
        action_inputs = self_vla.action_head.prepare_input(inputs)
        target_device = torch.device("cuda")
        target_dtype = self_vla.action_head.dtype

        def to_cuda(x):
            if torch.is_floating_point(x):
                return x.to(target_device, dtype=target_dtype)
            return x.to(target_device)

        backbone_inputs = _tree.map_structure(to_cuda, backbone_inputs)
        action_inputs = _tree.map_structure(to_cuda, action_inputs)
        return backbone_inputs, action_inputs

    type(policy.trained_model).prepare_input = _cuda_prepare_input

    action_head = policy.trained_model.action_head
    _orig_encode_prompt = type(action_head).encode_prompt
    _orig_encode_image = type(action_head).encode_image
    state: dict[str, Any] = {
        "text_encode_calls": 0, "image_encode_calls": 0,
        "text_cache_hits": 0, "text_cache_misses": 0,
        "prompt_emb_cache": {},
    }
    TEXT_PER_CHUNK = 2

    def _prompt_cache_key(input_ids, attention_mask):
        try:
            ids = input_ids.detach().cpu().numpy().tobytes() if torch.is_tensor(input_ids) else bytes(input_ids)
        except Exception:
            ids = repr(input_ids).encode()
        try:
            mask = attention_mask.detach().cpu().numpy().tobytes() if torch.is_tensor(attention_mask) else bytes(attention_mask)
        except Exception:
            mask = repr(attention_mask).encode()
        return (ids, mask)

    def _encode_prompt_wrapped(self_ah, input_ids, attention_mask):
        # Stage B: cache prompt embeddings so subsequent identical-prompt
        # calls (chunks N>0 of the same episode) never need to rehome the
        # 4 GiB text encoder back to CUDA.
        cache = state["prompt_emb_cache"]
        key = _prompt_cache_key(input_ids, attention_mask)
        if key in cache:
            state["text_cache_hits"] += 1
            cached = cache[key]
            if torch.is_tensor(cached):
                return cached.detach().clone()
            return cached

        state["text_cache_misses"] += 1
        try:
            te_dev = next(self_ah.text_encoder.parameters()).device
        except StopIteration:
            te_dev = torch.device("cuda")
        if te_dev.type == "cpu":
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            self_ah.text_encoder.to("cuda")
        if torch.is_tensor(input_ids) and input_ids.device.type == "cpu":
            input_ids = input_ids.to("cuda", non_blocking=True)
        if torch.is_tensor(attention_mask) and attention_mask.device.type == "cpu":
            attention_mask = attention_mask.to("cuda", non_blocking=True)
        state["text_encode_calls"] += 1
        out = _orig_encode_prompt(self_ah, input_ids, attention_mask)
        if torch.is_tensor(out):
            cache[key] = out.detach().clone()
        if state["text_encode_calls"] % TEXT_PER_CHUNK == 0:
            try:
                self_ah.text_encoder.to("cpu")
                gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
                print(f"  [offload] text_encoder -> cpu (cache size={len(cache)})")
            except Exception as e:
                print(f"  [offload] text_encoder offload failed: {e!r}")
        return out

    def _encode_image_wrapped(self_ah, image, num_frames, height, width):
        try:
            ie_dev = next(self_ah.image_encoder.parameters()).device
        except StopIteration:
            ie_dev = torch.device("cuda")
        if ie_dev.type == "cpu":
            self_ah.image_encoder.to("cuda")
        if torch.is_tensor(image) and image.device.type == "cpu":
            image = image.to("cuda", non_blocking=True)
        state["image_encode_calls"] += 1
        out = _orig_encode_image(self_ah, image, num_frames, height, width)
        try:
            self_ah.image_encoder.to("cpu")
            gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
        except Exception as e:
            print(f"  [offload] image_encoder offload failed: {e!r}")
        return out

    type(action_head).encode_prompt = _encode_prompt_wrapped
    type(action_head).encode_image = _encode_image_wrapped
    return state


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    import numpy as np
    t_start = time.perf_counter()

    output_dir = Path(os.environ.get("OUTPUT_DIR", "/artifacts/stage5_long"))
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = os.environ["MODEL_PATH"]
    embodiment = os.environ.get("EMBODIMENT_TAG", "OXE_DROID")
    dataset_dir = Path(os.environ.get("DROID_DATASET_DIR", "/data/lerobot_droid_101"))
    episode_list_str = os.environ.get("EPISODES", "1,13,22,82")
    episode_list = [int(s.strip()) for s in episode_list_str.split(",") if s.strip()]
    num_chunks = int(os.environ.get("NUM_CHUNKS", "4"))
    fps = int(os.environ.get("VIDEO_FPS", "5"))
    anchor_local = int(os.environ.get("ANCHOR_LOCAL_FRAME", "23"))
    streaming_overlap_decode = os.environ.get(
        "STREAMING_OVERLAP_DECODE", "1"
    ).strip().lower() in ("1", "true", "yes", "on")
    streaming_overlap_k = int(os.environ.get("STREAMING_OVERLAP_K", "1"))

    # ---- Stage H full-episode eval modes -----------------------------------
    # AUTO_PER_EPISODE_K=1 : per-episode K_eff = max chunks that fit the
    #   episode honestly (no frame-index clamping, no looping). Each episode
    #   gets its OWN K based on its own length:
    #       K_eff = 1 + max(0, (ep_len - 1 - anchor) // 24 + 1)
    # AUTO_PICK_EPISODES=N : sample N episodes spanning the dataset's length
    #   distribution by quantile (10, 30, 50, 70, 90 percentiles); overrides
    #   the EPISODES env. Uses seed 42 for reproducibility.
    # SKIP_GROUNDED_VIDEO=1 : do NOT emit long_video.mp4 from the grounded
    #   pass (that's the one with chunk-boundary restart artifacts). Action
    #   chunks are still saved (grounded actions are the real-world useful
    #   thing). Use together with LONG_VIDEO_AUTOREGRESSIVE_DEBUG=1 for a
    #   continuous predicted video without the restart artifact.
    # MAX_PER_EPISODE_K=N : cap the per-episode K (useful to keep total wall
    #   manageable when picking long episodes). 0 = uncapped.
    auto_per_ep_k = os.environ.get(
        "AUTO_PER_EPISODE_K", "0"
    ).strip().lower() in ("1", "true", "yes", "on")
    auto_pick_n = int(os.environ.get("AUTO_PICK_EPISODES", "0") or "0")
    skip_grounded_video = os.environ.get(
        "SKIP_GROUNDED_VIDEO", "0"
    ).strip().lower() in ("1", "true", "yes", "on")
    max_per_ep_k = int(os.environ.get("MAX_PER_EPISODE_K", "0") or "0")
    fifo_action_mode = os.environ.get(
        "FIFO_ACTION_MODE", "0"
    ).strip().lower() in ("1", "true", "yes", "on")
    defer_video_decode = os.environ.get(
        "DEFER_VIDEO_DECODE", "1" if fifo_action_mode else "0"
    ).strip().lower() in ("1", "true", "yes", "on")

    banner("Stage 5 LONG-FORM multi-chunk renderer")
    print(f"[cfg] MODEL_PATH         = {model_path}")
    print(f"[cfg] DROID_DATASET_DIR  = {dataset_dir}")
    print(f"[cfg] OUTPUT_DIR         = {output_dir}")
    print(f"[cfg] EPISODES           = {episode_list}")
    print(f"[cfg] NUM_CHUNKS         = {num_chunks}")
    print(f"[cfg] VIDEO_FPS          = {fps}")
    print(f"[cfg] ANCHOR_LOCAL_FRAME = {anchor_local}")
    print(f"[cfg] STREAMING_OVERLAP_DECODE = {streaming_overlap_decode}  "
          f"(if 1: per-chunk videos use rolling latent overlap)")
    print(f"[cfg] STREAMING_OVERLAP_K      = {streaming_overlap_k}")
    print(f"[cfg] STAGE_B            = {_STAGE_B_ENABLED}  (max_chunk_size={_STAGE_B_MAX_CHUNK_SIZE})")
    print(f"[cfg] STAGE_C            = {_STAGE_C_ENABLED}  (encode_image chunk-0 cache)")
    print(f"[cfg] ENABLE_FLASH_SDPA  = {os.environ.get('ENABLE_FLASH_SDPA','0')}")
    print(f"[cfg] FLASH_ATTN_BACKEND = {os.environ.get('FLASH_ATTN_BACKEND','sdpa')}")
    print(f"[cfg] FLASH_ATTENTION_TRITON_AMD_ENABLE = "
          f"{os.environ.get('FLASH_ATTENTION_TRITON_AMD_ENABLE','FALSE')}")
    print(f"[cfg] DENOISE_STEPS      = {os.environ.get('DENOISE_STEPS','<default 16>')}")
    print(f"[cfg] LONG_VIDEO_AUTOREGRESSIVE_DEBUG = {_LONG_VIDEO_AUTOREGRESSIVE_DEBUG}  "
          f"(if 1: extra ~K-chunk pass emits long_video_autoregressive_debug.mp4)")
    print(f"[cfg] DECOUPLE_INFERENCE_NOISE      = {os.environ.get('DECOUPLE_INFERENCE_NOISE','0')}  "
          f"(Stage F; video sigmas end at VIDEO_INFERENCE_FINAL_NOISE={os.environ.get('VIDEO_INFERENCE_FINAL_NOISE','<upstream 0.8>')})")
    print(f"[cfg] AUTO_PER_EPISODE_K            = {auto_per_ep_k}  "
          f"(K_eff computed per-episode from ep_len; overrides NUM_CHUNKS)")
    print(f"[cfg] AUTO_PICK_EPISODES            = {auto_pick_n}  "
          f"(if N>0: sample N episodes by length quantile; overrides EPISODES)")
    print(f"[cfg] SKIP_GROUNDED_VIDEO           = {skip_grounded_video}  "
          f"(suppress long_video.mp4 from grounded pass; keeps action_chunks.npz)")
    print(f"[cfg] MAX_PER_EPISODE_K             = {max_per_ep_k}  "
          f"(cap per-episode K; 0=uncapped)")
    print(f"[cfg] FIFO_ACTION_MODE             = {fifo_action_mode}  "
          f"(if 1: save action chunk immediately after model inference)")
    print(f"[cfg] DEFER_VIDEO_DECODE           = {defer_video_decode}  "
          f"(if 1: decode/video artifacts run after action chunks are emitted)")

    mem_trace: list[dict[str, Any]] = []

    def probe(label: str) -> None:
        h = host_mem_gib(); c = cuda_mem_gib()
        print(f"[mem] {label:<32} RAM used={h['used_gib']:.1f} GiB  "
              f"VRAM alloc={c['alloc_gib']:.2f} GiB  max={c['max_alloc_gib']:.2f} GiB  "
              f"reserved={c['reserved_gib']:.2f} GiB", flush=True)
        mem_trace.append({"label": label, "ts": time.perf_counter() - t_start, **h, **c})

    probe("00_script_start")

    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
    try:
        import torch
        import torch.distributed as dist
        import torch._dynamo
        torch._dynamo.config.disable = True
        torch._dynamo.config.suppress_errors = True
        torch._dynamo.reset()
    except Exception as exc:
        print(f"[FATAL] base import failed: {exc!r}")
        return 2

    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29503")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        dist.init_process_group(backend="gloo", rank=0, world_size=1)

    probe("01_post_torch_init")

    # ---- Model + dataset imports -------------------------------------------
    try:
        from groot.vla.model.n1_5.sim_policy import GrootSimPolicy
        from groot.vla.model.dreamzero.base_vla import VLA, VLAConfig
        from groot.vla.data.schema import EmbodimentTag
        from groot.vla.model.dreamzero.action_head.wan_flow_matching_action_tf import (
            WANPolicyHead as _WANPolicyHead,
        )
        from tianshou.data import Batch
        from safetensors.torch import load_file as _load_safetensors
        from validate_stage3_path_b import DroidDataset
    except Exception as exc:
        print(f"[FATAL] model/dataset import failed: {exc!r}")
        traceback.print_exc()
        return 3

    # ---- Patches that must be set BEFORE policy load -----------------------
    @classmethod
    def _amd_low_mem_from_pretrained(cls, pretrained_model_name_or_path, config=None):
        del config
        cfg_path = os.path.join(pretrained_model_name_or_path, "config.json")
        with open(cfg_path) as f:
            cfg_dict = json.load(f)
        if isinstance(cfg_dict.get("action_head_cfg", {}).get("config"), dict):
            cfg_dict["action_head_cfg"]["config"]["defer_lora_injection"] = False
        if _STAGE_B_ENABLED:
            try:
                from _stage_b_patches import patch_max_chunk_size_in_cfg
                patch_max_chunk_size_in_cfg(cfg_dict, value=_STAGE_B_MAX_CHUNK_SIZE)
            except Exception as exc:
                print(f"[stage-b] patch (2) skipped: {exc!r}")
        cfg = VLAConfig(**cfg_dict)
        prev_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            t0 = time.perf_counter()
            model = cls(cfg)
            print(f"[low-mem] empty bf16 model built in {time.perf_counter()-t0:.1f} s")
        finally:
            torch.set_default_dtype(prev_dtype)

        st_idx = os.path.join(pretrained_model_name_or_path, "model.safetensors.index.json")
        st_one = os.path.join(pretrained_model_name_or_path, "model.safetensors")
        missing, unexpected = set(), set()

        def _apply(sd):
            if any(".base_layer." in k for k in sd):
                sd = {k.replace(".base_layer.", "."): v for k, v in sd.items()}
            mk, uk = model.load_state_dict(sd, strict=False)
            return mk, uk

        if os.path.exists(st_idx):
            with open(st_idx) as f:
                index = json.load(f)
            shards = sorted(set(index["weight_map"].values()))
            for i, shard in enumerate(shards):
                t0 = time.perf_counter()
                sd = _load_safetensors(os.path.join(pretrained_model_name_or_path, shard))
                mk, uk = _apply(sd)
                missing.update(mk); unexpected.update(uk)
                del sd; gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"[low-mem]   shard {i+1}/{len(shards)} {shard} in {time.perf_counter()-t0:.1f}s")
        elif os.path.exists(st_one):
            sd = _load_safetensors(st_one)
            mk, uk = _apply(sd); missing.update(mk); unexpected.update(uk)
            del sd; gc.collect()
        else:
            raise FileNotFoundError(f"No safetensors at {pretrained_model_name_or_path}")
        if missing:
            print(f"[low-mem] missing keys: {len(missing)}")
        if unexpected:
            print(f"[low-mem] unexpected keys: {len(unexpected)}")
        return model

    VLA.from_pretrained = _amd_low_mem_from_pretrained

    def _amd_post_initialize(self_ah):
        self_ah.model.to(device=self_ah._device, dtype=torch.bfloat16)
        self_ah.text_encoder.to(device=self_ah._device, dtype=torch.bfloat16)
        self_ah.image_encoder.to(device=self_ah._device, dtype=torch.bfloat16)
        self_ah.vae.to(device=self_ah._device, dtype=torch.bfloat16)
        self_ah.trt_engine = None
    _WANPolicyHead.post_initialize = _amd_post_initialize

    # ---- Open dataset ------------------------------------------------------
    banner("Opening DROID dataset")
    try:
        ds = DroidDataset(dataset_dir)
    except Exception as exc:
        print(f"[FATAL] DroidDataset open failed: {exc!r}")
        traceback.print_exc()
        return 3
    # Validate episodes: require length >= 23 + 24*K + 1 to fit K-chunk schedule.
    # Stage G K-scaling bench (BENCH_K_OVERRIDE_LENGTH_CHECK=1) disables this
    # prefilter and lets build_long_schedule clamp frame indices to ep_len-1,
    # which effectively repeats the last frame for chunks past episode end.
    # Used to probe pure K-scaling memory + wall behavior without needing a
    # 3000-frame DROID episode (none exists).
    bench_k_override = os.environ.get(
        "BENCH_K_OVERRIDE_LENGTH_CHECK", "0"
    ).strip().lower() in ("1", "true", "yes", "on")

    # Stage H full-episode eval: auto-pick N episodes by length quantile.
    # Samples ~300 random episodes, queries their lengths, sorts and picks
    # at the 10/30/50/70/90 percentiles within a sane working range. Each
    # picked episode then runs at its OWN K_eff (computed in the per-episode
    # loop below). Reproducible via random.seed(42).
    if auto_pick_n > 0:
        try:
            import random as _random
            _random.seed(42)
            try:
                n_total = len(ds)
            except Exception:
                n_total = 95000  # fallback; DROID 101 has ~95.6k episodes
            sample_n = min(300, n_total)
            cand_ids = _random.sample(range(n_total), sample_n)
            cand_info = []
            for cid in cand_ids:
                try:
                    cinfo = ds.episode_info(cid)
                    cand_info.append((cid, int(cinfo["length"]), cinfo.get("task") or ""))
                except Exception:
                    continue
            # Working range: skip very short (<100 frames) and very long
            # (>900 frames). Very-short ones produce K_eff < 4 which is not
            # a meaningful rollout; very long ones blow up the wall budget.
            filtered = [t for t in cand_info if 100 < t[1] < 900]
            filtered.sort(key=lambda t: t[1])
            print(f"[auto-pick] sampled {len(cand_info)} eps; "
                  f"{len(filtered)} in working range [100, 900) frames")
            if len(filtered) < auto_pick_n:
                print(f"[auto-pick] FATAL: not enough candidates in range "
                      f"({len(filtered)} < {auto_pick_n})")
                return 3
            quantiles = [int(len(filtered) * q) for q in
                         [0.1, 0.3, 0.5, 0.7, 0.9][:auto_pick_n]]
            quantiles = [min(i, len(filtered) - 1) for i in quantiles]
            picked = [filtered[i] for i in quantiles]
            episode_list = [p[0] for p in picked]
            print(f"[auto-pick] EPISODES = {episode_list}")
            for ep_id, ln, task in picked:
                if ln <= anchor_local + 1:
                    K_eff = 1
                else:
                    K_eff = 1 + max(0, (ln - 1 - anchor_local) // 24 + 1)
                if max_per_ep_k > 0:
                    K_eff = min(K_eff, max_per_ep_k)
                print(f"[auto-pick]   ep {ep_id:>5}  len={ln:>4}  "
                      f"K_eff={K_eff:>3}  task={task!r}")
        except Exception as exc:
            print(f"[auto-pick] FATAL: {exc!r}")
            traceback.print_exc()
            return 3

    valid_episodes: list[int] = []
    for e in episode_list:
        try:
            info = ds.episode_info(e)
        except Exception as exc:
            print(f"[skip-presence] ep {e}: {exc!r}")
            continue
        if auto_per_ep_k:
            # Per-episode K mode: any episode with at least 1 honest grounded
            # chunk (K_eff >= 2) is valid. Per-episode K_eff is computed in
            # the per-episode loop and clamps NUM_CHUNKS for that episode.
            if info["length"] < anchor_local + 1:
                print(f"[skip-too-short-for-anchor] ep {e} length={info['length']} "
                      f"(need >= {anchor_local + 1} for anchor at frame {anchor_local})")
                continue
            valid_episodes.append(e)
            print(f"  ep {e:>4}  len={info['length']:>4}  task={info['task']!r}")
            continue

        min_len = 23 + 24 * num_chunks + 1
        if info["length"] < min_len:
            if bench_k_override:
                print(f"[bench-k-allow-short] ep {e} length={info['length']} < min_len={min_len}; "
                      f"frame indices will clamp to ep_len-1 for late chunks")
            else:
                print(f"[skip-too-short] ep {e} length={info['length']} (need >= {min_len} for K={num_chunks})")
                continue
        valid_episodes.append(e)
        print(f"  ep {e:>4}  len={info['length']:>4}  task={info['task']!r}")
    print(f"[ok] dataset opened; {len(valid_episodes)}/{len(episode_list)} episodes valid")
    if not valid_episodes:
        print("[FATAL] no valid episodes")
        return 3

    # ---- Policy load -------------------------------------------------------
    banner("Loading GrootSimPolicy (one-time)")
    probe("03_pre_policy_load")
    t_load = time.perf_counter()
    try:
        policy = GrootSimPolicy(
            embodiment_tag=getattr(EmbodimentTag, embodiment),
            model_path=model_path,
            device="cuda",
        )
    except Exception as exc:
        print(f"[FATAL] policy load failed: {exc!r}")
        traceback.print_exc()
        return 4
    load_dt = time.perf_counter() - t_load
    print(f"[ok] policy loaded in {load_dt:.1f} s")
    probe("04_post_policy_load")

    _stage_b_state = apply_amd_patches(policy)
    try:
        policy._stage_b_state = _stage_b_state
    except Exception:
        pass
    probe("05_post_patches")

    if _STAGE_B_ENABLED:
        try:
            from _stage_b_patches import apply_stage_b_post_load_patches
            apply_stage_b_post_load_patches(policy)
            probe("05b_post_stage_b_patches")
        except Exception as exc:
            print(f"[FATAL] Stage B post-load patches failed: {exc!r}")
            traceback.print_exc()
            return 4

    if _STAGE_C_ENABLED:
        try:
            from _stage_c_patches import apply_stage_c_patch
            apply_stage_c_patch(policy)
            probe("05c_post_stage_c_patch")
        except Exception as exc:
            print(f"[FATAL] Stage C patch failed: {exc!r}")
            traceback.print_exc()
            return 4

    if os.environ.get("ENABLE_FLASH_SDPA", "0") == "1":
        os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
            print("[perf] flash + mem-efficient SDPA enabled")
        except Exception as e:
            print(f"[perf] enable_flash_sdp failed: {e!r}")
    denoise_override = os.environ.get("DENOISE_STEPS", "").strip()
    if denoise_override:
        try:
            n = int(denoise_override)
            old = int(getattr(policy.trained_model.action_head, "num_inference_steps", 16))
            policy.trained_model.action_head.num_inference_steps = n
            print(f"[perf] num_inference_steps {old} -> {n}")
        except Exception as e:
            print(f"[perf] DENOISE_STEPS override failed: {e!r}")

    # ---- Stage F: decoupled inference noise (video stays noisy, action denoises fully) ----
    # See validate_stage3_path_a.py for the long form; same upstream config
    # fields. Set DECOUPLE_INFERENCE_NOISE=1 to enable, optionally override
    # VIDEO_INFERENCE_FINAL_NOISE (default upstream 0.8).
    decouple_env = os.environ.get("DECOUPLE_INFERENCE_NOISE", "").strip().lower()
    if decouple_env in ("1", "true", "yes", "on"):
        try:
            cfg = policy.trained_model.action_head.config
            old_b = bool(getattr(cfg, "decouple_inference_noise", False))
            old_v = float(getattr(cfg, "video_inference_final_noise", 0.8))
            cfg.decouple_inference_noise = True
            v_env = os.environ.get("VIDEO_INFERENCE_FINAL_NOISE", "").strip()
            if v_env:
                cfg.video_inference_final_noise = float(v_env)
            new_v = float(cfg.video_inference_final_noise)
            print(f"[stage-f] decouple_inference_noise {old_b} -> True; "
                  f"video_inference_final_noise {old_v} -> {new_v}")
        except Exception as e:
            print(f"[stage-f] config override failed: {e!r}")

    # ---- Per-episode long-form loop ---------------------------------------
    if auto_per_ep_k:
        banner(f"Rendering {len(valid_episodes)} episodes "
               f"x K_eff (per-episode auto, anchor={anchor_local}, "
               f"max_cap={max_per_ep_k or 'none'})")
    else:
        banner(f"Rendering {len(valid_episodes)} episodes x K={num_chunks} chunks")
    cuda_mem_gib(reset=True)
    per_ep_rows: list[dict[str, Any]] = []
    successes = 0
    failures: list[dict[str, Any]] = []

    for ep_idx, ep in enumerate(valid_episodes):
        ep_tag = f"ep{ep:04d}"
        ep_dir = output_dir / ep_tag
        ep_dir.mkdir(parents=True, exist_ok=True)

        # Between-episode VRAM purge. CORRECT upstream attribute names from
        # wan_flow_matching_action_tf.WANPolicyHead.__init__:
        #   self.kv_cache1     (the rolling DiT KV cache, ~5 GiB after K=4)
        #   self.kv_cache_neg  (CFG negative branch)
        #   self.clip_feas, self.ys  (image-encoder cached features)
        #   self.current_start_frame: int  (streaming-causal position)
        #   self.language      (cached UMT5 embedding for the current prompt)
        # The older list ("kv_cache", "_kv_cache", "cached_clip_feas",
        # "cached_ys") was a no-op -- none of those attributes exist.
        if ep_idx > 0:
            pre_mem = (torch.cuda.memory_allocated() / 1024**3) if torch.cuda.is_available() else -1.0
            try:
                ah = policy.trained_model.action_head
                if hasattr(ah, "text_encoder"):
                    try:
                        ah.text_encoder.cpu()
                    except Exception:
                        pass
                if hasattr(ah, "vae"):
                    try:
                        vae_dev = next(ah.vae.parameters()).device.type
                        if vae_dev != "cuda":
                            ah.vae.to(device=torch.device("cuda"), dtype=torch.bfloat16)
                            print(f"[mem-purge] rehomed VAE cpu -> cuda")
                    except Exception as e:
                        print(f"[mem-purge] vae rehome check failed: {e!r}")
                for _attr in ("language", "clip_feas", "ys",
                              "kv_cache1", "kv_cache_neg"):
                    if hasattr(ah, _attr):
                        try:
                            setattr(ah, _attr, None)
                        except Exception:
                            pass
                if hasattr(ah, "current_start_frame"):
                    try:
                        ah.current_start_frame = 0
                    except Exception:
                        pass
                # Stage C: invalidate the episode-scoped chunk-0 encode_image
                # cache. Each episode has its own anchor frame so the cached
                # (clip_feas, ys, anchor_latent) from ep N must not leak
                # into ep N+1.
                if _STAGE_C_ENABLED:
                    try:
                        from _stage_c_patches import invalidate_stage_c_cache
                        invalidate_stage_c_cache(ah)
                    except Exception:
                        if hasattr(ah, "_stage_c_cache"):
                            try:
                                ah._stage_c_cache = None
                            except Exception:
                                pass
                # Drop our prompt-emb cache too: each episode has a different
                # task prompt, so old entries are useless and just hold ~10 MB
                # of UMT5 output tensors on CUDA.
                if hasattr(policy, "_stage_b_state"):
                    try:
                        policy._stage_b_state["prompt_emb_cache"].clear()
                    except Exception:
                        pass
            except Exception as e:
                print(f"[mem-purge] partial failure: {e!r}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                cm = torch.cuda.memory_allocated() / 1024**3
                print(f"[mem-purge] pre-ep {ep}: VRAM allocated "
                      f"{pre_mem:.2f} -> {cm:.2f} GiB", flush=True)

        info = ds.episode_info(ep)
        ep_len = info["length"]
        prompt = info["task"] or "Perform the demonstrated task."
        anchor = max(min(anchor_local, ep_len - 25), 23)

        # Stage H per-episode K: compute K_eff so the schedule fits the
        # episode honestly (no clamping, no looping). The grounded pass
        # needs anchor frames within real GT, so:
        #   K_eff = 1 + max(0, (ep_len - 1 - anchor) // 24 + 1)
        # MAX_PER_EPISODE_K can cap this when total wall budget matters.
        if auto_per_ep_k:
            if ep_len <= anchor + 1:
                K_eff = 1
            else:
                K_eff = 1 + max(0, (ep_len - 1 - anchor) // 24 + 1)
            if max_per_ep_k > 0:
                K_eff = min(K_eff, max_per_ep_k)
            num_chunks_this_ep = max(1, K_eff)
            print(f"\n[{ep_tag}] len={ep_len} prompt={prompt!r}")
            print(f"[{ep_tag}] AUTO_PER_EPISODE_K: K_eff={num_chunks_this_ep} "
                  f"(anchor={anchor}, schedule covers frames "
                  f"[0..{anchor + 24 * (num_chunks_this_ep - 2)}] honestly)")
        else:
            num_chunks_this_ep = num_chunks
            print(f"\n[{ep_tag}] len={ep_len} prompt={prompt!r}")

        schedule = build_long_schedule(anchor, ep_len, num_chunks_this_ep)
        K = len(schedule)
        print(f"[{ep_tag}] K={K} schedule:")
        for i, idx in enumerate(schedule):
            print(f"    chunk {i:>2}: {idx}")

        # Flag clamped chunks explicitly. With BENCH_K_OVERRIDE_LENGTH_CHECK=1
        # short episodes are accepted at high K, and build_long_schedule
        # clamps frame indices to ep_len-1 once the schedule overruns. Chunks
        # past the first fully-clamped index all receive identical inputs ->
        # the model emits bit-identical predicted latents -> the unified
        # decode shows the same ~5-frame segment looping. See
        # docs/STAGE_G_BENCH_LOOPING.md.
        last_real_idx = ep_len - 1
        clamped_chunks = [
            ci for ci, idx in enumerate(schedule)
            if all(f >= last_real_idx for f in idx)
        ]
        partial_clamped = [
            ci for ci, idx in enumerate(schedule)
            if any(f >= last_real_idx for f in idx) and ci not in clamped_chunks
        ]
        if clamped_chunks or partial_clamped:
            n_full = len(clamped_chunks)
            n_part = len(partial_clamped)
            first_full = clamped_chunks[0] if clamped_chunks else None
            pct_full = (100.0 * n_full / max(K, 1))
            print(f"[{ep_tag}] *** WARNING: schedule overruns episode "
                  f"(ep_len={ep_len}) ***")
            if partial_clamped:
                print(f"[{ep_tag}]   chunks with partial last-frame clamping: "
                      f"{n_part} (e.g. {partial_clamped[:5]})")
            if clamped_chunks:
                print(f"[{ep_tag}]   chunks with ALL frames clamped to "
                      f"{last_real_idx}: {n_full} ({pct_full:.0f}% of K), "
                      f"first fully-clamped chunk = {first_full}")
                print(f"[{ep_tag}]   --> chunks {first_full}..{K-1} will see "
                      f"identical inputs, predict identical futures, and "
                      f"produce a visually-looping unified_video. This is "
                      f"NOT a bug; see docs/STAGE_G_BENCH_LOOPING.md.")
                print(f"[{ep_tag}]   for honest long-horizon visualization "
                      f"set LONG_VIDEO_AUTOREGRESSIVE_DEBUG=1 to use "
                      f"autoregressive video extrapolation instead.")

        per_chunk_frames: list = []  # list of (T, H, W, 3) uint8 RGB tensors
        per_chunk_action: list = []  # list of (24, 8) float32
        per_chunk_meta: list[dict[str, Any]] = []
        fifo_action_events: list[dict[str, Any]] = []
        overlap_decode_prev_tail = None
        # A/B for the "rewind every chunk boundary" artifact: collect each
        # chunk's CPU-resident video_pred latent so we can do a single
        # unified VAE decode at end of episode (mirrors upstream
        # socket_test_optimized_AR.py::_reset_state, which torch.cats
        # video_across_time on dim=2 and decodes once).
        per_chunk_latents_cpu: list = []  # list of (1, 16, T_latent, H_lat, W_lat) bf16 on CPU
        anchor_rgb_for_sxs = None
        ep_failed = False

        for c_idx, frame_indices in enumerate(schedule):
            chunk_t0 = time.perf_counter()
            try:
                frames_dict = ds.read_frames(ep, frame_indices)
            except Exception as exc:
                print(f"[{ep_tag} chunk {c_idx}] read_frames failed: {exc!r}")
                failures.append({"episode": ep, "chunk": c_idx, "stage": "read_frames", "error": repr(exc)})
                ep_failed = True
                break

            if c_idx == 0:
                try:
                    anchor_rgb_for_sxs = frames_dict[PRIMARY_NATIVE_CAM][-1]
                except Exception:
                    pass

            obs: dict[str, Any] = dict(frames_dict)
            try:
                anchor_state_8 = ds.read_state(ep, frame_indices[-1])
                if anchor_state_8.shape == (8,):
                    js = anchor_state_8[:7].reshape(1, 7).astype(np.float64)
                    gs = anchor_state_8[7:8].reshape(1, 1).astype(np.float64)
                elif anchor_state_8.shape == (7,):
                    js = anchor_state_8.reshape(1, 7).astype(np.float64)
                    gs = np.zeros((1, 1), dtype=np.float64)
                else:
                    raise RuntimeError(f"unexpected state shape {anchor_state_8.shape}")
            except Exception as exc:
                print(f"[{ep_tag} chunk {c_idx}] state read failed: {exc!r}; zeros")
                js = np.zeros((1, 7), dtype=np.float64)
                gs = np.zeros((1, 1), dtype=np.float64)
            obs["state.joint_position"] = js
            obs["state.gripper_position"] = gs
            obs["annotation.language.action_text"] = prompt

            try:
                batch_in = Batch(obs=obs)
                result = policy.lazy_joint_forward_causal(batch_in)
                torch.cuda.synchronize()
            except Exception as exc:
                print(f"[{ep_tag} chunk {c_idx}] inference failed: {exc!r}")
                traceback.print_exc()
                failures.append({"episode": ep, "chunk": c_idx, "stage": "inference", "error": repr(exc)})
                ep_failed = True
                # Recovery: drop chunk-state so next ep can start clean.
                try:
                    ah = policy.trained_model.action_head
                    for _attr in ("language", "clip_feas", "ys",
                                  "kv_cache1", "kv_cache_neg"):
                        if hasattr(ah, _attr):
                            try:
                                setattr(ah, _attr, None)
                            except Exception:
                                pass
                    if hasattr(ah, "current_start_frame"):
                        try:
                            ah.current_start_frame = 0
                        except Exception:
                            pass
                    if hasattr(ah, "_stage_c_cache"):
                        try:
                            ah._stage_c_cache = None
                        except Exception:
                            pass
                    if hasattr(policy, "_stage_b_state"):
                        try:
                            policy._stage_b_state["prompt_emb_cache"].clear()
                        except Exception:
                            pass
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                except Exception:
                    pass
                if c_idx == 0 and ep_idx == 0 and successes == 0:
                    return 5
                break

            infer_dt = time.perf_counter() - chunk_t0
            if not isinstance(result, tuple) or len(result) < 2:
                print(f"[{ep_tag} chunk {c_idx}] result missing video_pred")
                failures.append({"episode": ep, "chunk": c_idx, "stage": "result_shape", "error": "no video_pred"})
                ep_failed = True
                break
            batch_out, video_pred = result[0], result[1]
            if video_pred is None:
                print(f"[{ep_tag} chunk {c_idx}] video_pred is None")
                failures.append({"episode": ep, "chunk": c_idx, "stage": "video_pred_none", "error": ""})
                ep_failed = True
                break

            pred_action = extract_action_chunk(batch_out)
            action_ready_dt = time.perf_counter() - chunk_t0
            if pred_action is not None:
                per_chunk_action.append(pred_action)
                if fifo_action_mode:
                    try:
                        action_path = ep_dir / f"action_chunk_{c_idx:02d}.npy"
                        np.save(action_path, pred_action)
                        event = {
                            "episode": ep,
                            "chunk": c_idx,
                            "frames": list(frame_indices),
                            "action_ready_s": round(action_ready_dt, 3),
                            "action_shape": list(pred_action.shape),
                            "action_path": str(action_path),
                            "ts_since_script_start_s": round(time.perf_counter() - t_start, 3),
                        }
                        fifo_action_events.append(event)
                        with open(ep_dir / "action_fifo.jsonl", "a") as f:
                            f.write(json.dumps(event) + "\n")
                        print(
                            f"[{ep_tag} chunk {c_idx}] FIFO_ACTION_READY "
                            f"action_ready={action_ready_dt:.3f}s  "
                            f"shape={tuple(pred_action.shape)}  "
                            f"path={action_path.name}",
                            flush=True,
                        )
                    except Exception as exc:
                        print(f"[{ep_tag} chunk {c_idx}] FIFO action emit failed: {exc!r}")

            try:
                per_chunk_latents_cpu.append(
                    video_pred.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
                )
            except Exception as e:
                print(f"[{ep_tag} chunk {c_idx}] latent cpu snapshot failed: {e!r}")

            video_pred_shape = list(video_pred.shape)
            action_shape = (list(pred_action.shape) if pred_action is not None else None)

            if defer_video_decode:
                decode_dt = 0.0
                decode_meta = {
                    "mode": "deferred",
                    "k_overlap": int(streaming_overlap_k if streaming_overlap_decode else 0),
                    "decode_input_latent_frames": int(video_pred.shape[2]),
                    "skipped_rgb_frames": 0,
                    "emitted_rgb_frames": 0,
                }
                frames_u8 = None
            else:
                try:
                    if streaming_overlap_decode:
                        frames_u8, decode_dt, overlap_decode_prev_tail, decode_meta = (
                            decode_video_pred_overlap(
                                policy=policy,
                                video_pred=video_pred,
                                prev_tail=overlap_decode_prev_tail,
                                k_overlap=streaming_overlap_k,
                            )
                        )
                    else:
                        frames_u8, decode_dt = decode_video_pred(policy, video_pred)
                        decode_meta = {
                            "mode": "isolated",
                            "k_overlap": 0,
                            "decode_input_latent_frames": int(video_pred.shape[2]),
                            "skipped_rgb_frames": 0,
                            "emitted_rgb_frames": int(frames_u8.shape[0]),
                        }
                except Exception as exc:
                    print(f"[{ep_tag} chunk {c_idx}] VAE decode failed: {exc!r}")
                    traceback.print_exc()
                    failures.append({"episode": ep, "chunk": c_idx, "stage": "vae_decode", "error": repr(exc)})
                    ep_failed = True
                    break

            if frames_u8 is not None:
                per_chunk_frames.append(frames_u8)
                try:
                    write_mp4(ep_dir / f"per_chunk_video_{c_idx:02d}.mp4", frames_u8, fps)
                except Exception as e:
                    print(f"[{ep_tag} chunk {c_idx}] per-chunk mp4 write failed: {e!r}")

            # Stage B: drop chunk-local GPU references so chunk N+1 has
            # ~340 MiB more headroom for the next encode_prompt / vae.encode.
            try:
                del result, batch_out, video_pred, pred_action
            except Exception:
                pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            mem = cuda_mem_gib()
            host = host_mem_gib()
            # Stage G K-scaling bench: track cumulative CPU latent buffer
            # bytes so the K-scaling curve includes both VRAM and CPU RAM.
            latent_buf_mib = sum(
                l.element_size() * l.numel() for l in per_chunk_latents_cpu
            ) / (1024 ** 2)
            per_chunk_meta.append({
                "chunk": c_idx,
                "frames": list(frame_indices),
                "infer_s": round(infer_dt, 2),
                "decode_s": round(decode_dt, 2),
                "action_ready_s": round(action_ready_dt, 2),
                "fifo_action_mode": bool(fifo_action_mode),
                "decode_deferred": bool(defer_video_decode),
                "decode_mode": decode_meta["mode"],
                "decode_k_overlap": decode_meta["k_overlap"],
                "decode_input_latent_frames": decode_meta["decode_input_latent_frames"],
                "decode_skipped_rgb_frames": decode_meta["skipped_rgb_frames"],
                "video_pred_shape": video_pred_shape,
                "decoded_rgb_shape": list(frames_u8.shape) if frames_u8 is not None else [0, 0, 0, 0],
                "action_shape": action_shape,
                "vram_alloc_gib": round(mem["alloc_gib"], 3),
                "vram_max_alloc_gib": round(mem["max_alloc_gib"], 3),
                "host_ram_used_gib": round(host["used_gib"], 3),
                "latent_buffer_mib": round(latent_buf_mib, 3),
                "latent_buffer_count": len(per_chunk_latents_cpu),
            })
            print(f"[{ep_tag} chunk {c_idx}] frames={frame_indices}  "
                  f"infer={infer_dt:.1f}s  decode={decode_dt:.1f}s  "
                  f"action_ready={action_ready_dt:.1f}s  "
                  f"video_pred={tuple(video_pred_shape)} -> "
                  f"rgb={tuple(frames_u8.shape) if frames_u8 is not None else '<deferred>'}  "
                  f"decode_mode={decode_meta['mode']}  "
                  f"VRAM={mem['max_alloc_gib']:.1f} GiB  "
                  f"host_RAM={host['used_gib']:.1f} GiB  "
                  f"latent_buf={latent_buf_mib:.1f} MiB ({len(per_chunk_latents_cpu)} chunks)",
                  flush=True)

        # End of per-chunk loop. FIFO mode defers visual diagnostics until all
        # action chunks have already been emitted, keeping decode off the
        # critical action path.
        if defer_video_decode and per_chunk_latents_cpu and not per_chunk_frames:
            print(f"[{ep_tag}] FIFO_ACTION_MODE: starting deferred VAE decode "
                  f"for {len(per_chunk_latents_cpu)} chunks after action emission")
            overlap_decode_prev_tail = None
            for i, lat_cpu in enumerate(per_chunk_latents_cpu):
                try:
                    if streaming_overlap_decode:
                        frames_u8, decode_dt, overlap_decode_prev_tail, decode_meta = (
                            decode_video_pred_overlap(
                                policy=policy,
                                video_pred=lat_cpu,
                                prev_tail=overlap_decode_prev_tail,
                                k_overlap=streaming_overlap_k,
                            )
                        )
                    else:
                        frames_u8, decode_dt = decode_video_pred(policy, lat_cpu)
                        decode_meta = {
                            "mode": "isolated",
                            "k_overlap": 0,
                            "decode_input_latent_frames": int(lat_cpu.shape[2]),
                            "skipped_rgb_frames": 0,
                            "emitted_rgb_frames": int(frames_u8.shape[0]),
                        }
                    per_chunk_frames.append(frames_u8)
                    if i < len(per_chunk_meta):
                        per_chunk_meta[i].update({
                            "decode_s": round(decode_dt, 2),
                            "decode_mode": decode_meta["mode"],
                            "decode_k_overlap": decode_meta["k_overlap"],
                            "decode_input_latent_frames": decode_meta["decode_input_latent_frames"],
                            "decode_skipped_rgb_frames": decode_meta["skipped_rgb_frames"],
                            "decoded_rgb_shape": list(frames_u8.shape),
                        })
                    try:
                        write_mp4(ep_dir / f"per_chunk_video_{i:02d}.mp4", frames_u8, fps)
                    except Exception as exc:
                        print(f"[{ep_tag} chunk {i}] deferred per-chunk mp4 write failed: {exc!r}")
                    print(f"[{ep_tag} chunk {i}] FIFO_DEFERRED_DECODE "
                          f"decode={decode_dt:.1f}s  rgb={tuple(frames_u8.shape)}  "
                          f"decode_mode={decode_meta['mode']}", flush=True)
                except Exception as exc:
                    print(f"[{ep_tag} chunk {i}] deferred VAE decode failed: {exc!r}")
                    traceback.print_exc()
                    failures.append({"episode": ep, "chunk": i, "stage": "deferred_vae_decode", "error": repr(exc)})
                    ep_failed = True
                    break

        # End of per-chunk loop. Emit long video / artifacts if we got at least 1 chunk.
        if not per_chunk_frames and not per_chunk_action:
            continue

        # ----- Main long_video.mp4 = unified VAE decode of all per-chunk
        # observation-grounded latents (mirrors upstream
        # socket_test_optimized_AR.py::_reset_state). Produces (N_lat-1)*4+1
        # RGB frames in a single decode call, avoiding the temporal-stride-4
        # "first-frame" artifact that comes from decoding each chunk
        # independently. NOTE: this still contains the *intrinsic* model-level
        # "restart" at chunk boundaries -- chunk N's predicted tail is the
        # model's extrapolation from its anchor, chunk N+1's predicted start
        # is grounded on fresh observation frames at frame_indices[N+1], so
        # the latent stream itself is discontinuous at the boundary. That is
        # the production action-grounded behavior. For a continuous predicted-
        # video debug artifact see long_video_autoregressive_debug.mp4 below.
        long_rgb = None
        unified_decode_stats = {
            "attempted": False,
            "succeeded": False,
            "wall_s": None,
            "vram_peak_during_gib": None,
            "host_ram_peak_during_gib": None,
            "n_latent_frames": None,
            "expected_rgb_frames": None,
            "err": None,
        }
        if skip_grounded_video:
            print(f"[{ep_tag}] SKIP_GROUNDED_VIDEO=1: skipping unified decode "
                  f"and long_video.mp4 / long_video_chunkwise_debug.mp4 "
                  f"(the grounded-mode videos have chunk-boundary 'restart' "
                  f"artifacts; AR-mode video below is the continuous artifact)")
            unified_decode_stats["err"] = "skipped (SKIP_GROUNDED_VIDEO=1)"
        else:
            try:
                if per_chunk_latents_cpu:
                    # Stage G K-scaling bench: instrument peak VRAM/host RAM
                    # across the unified decode so we can see when this step
                    # becomes the wall (it's the single biggest GPU allocation
                    # for long episodes).
                    if torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats()
                    host_pre = host_mem_gib()["used_gib"]
                    latent_cat = torch.cat(per_chunk_latents_cpu, dim=2)
                    n_lat = latent_cat.shape[2]
                    expected_rgb = max(0, (n_lat - 1) * 4 + 1)
                    unified_decode_stats["attempted"] = True
                    unified_decode_stats["n_latent_frames"] = int(n_lat)
                    unified_decode_stats["expected_rgb_frames"] = int(expected_rgb)
                    t_dec0 = time.perf_counter()
                    long_rgb, _ = decode_video_pred(policy, latent_cat)
                    t_dec = time.perf_counter() - t_dec0
                    host_post = host_mem_gib()["used_gib"]
                    vram_peak = cuda_mem_gib()["max_alloc_gib"]
                    unified_decode_stats["succeeded"] = True
                    unified_decode_stats["wall_s"] = round(t_dec, 2)
                    unified_decode_stats["vram_peak_during_gib"] = round(vram_peak, 3)
                    unified_decode_stats["host_ram_peak_during_gib"] = round(
                        max(host_pre, host_post), 3
                    )
                    write_mp4(ep_dir / "long_video.mp4", long_rgb, fps)
                    print(f"[{ep_tag}] saved long_video.mp4  "
                          f"shape={long_rgb.shape}  "
                          f"duration~{long_rgb.shape[0] / fps:.1f}s  "
                          f"latent_cat=({n_lat} t-frames, expected_rgb={expected_rgb})  "
                          f"unified_decode={t_dec:.1f}s  "
                          f"vram_peak={vram_peak:.2f} GiB  "
                          f"host_ram_peak={max(host_pre, host_post):.1f} GiB")
                else:
                    long_rgb = np.concatenate(per_chunk_frames, axis=0)
                    write_mp4(ep_dir / "long_video.mp4", long_rgb, fps)
                    print(f"[{ep_tag}] saved long_video.mp4 (chunkwise fallback)  "
                          f"shape={long_rgb.shape}")
            except Exception as e:
                unified_decode_stats["err"] = repr(e)
                print(f"[{ep_tag}] long_video.mp4 unified decode failed: {e!r}")
                traceback.print_exc()
                try:
                    long_rgb = np.concatenate(per_chunk_frames, axis=0)
                    write_mp4(ep_dir / "long_video.mp4", long_rgb, fps)
                except Exception:
                    pass

        # ----- Streaming per-chunk decode stitched mp4. With
        # STREAMING_OVERLAP_DECODE=1 this is the production-style fixed stream:
        # per chunk, causal, but each decode receives the previous latent tail
        # so the VAE's temporal convs do not restart at chunk boundaries.
        streaming_rgb = None
        streaming_decode_stats = {
            "enabled": bool(streaming_overlap_decode),
            "k_overlap": int(streaming_overlap_k if streaming_overlap_decode else 0),
            "frames": int(sum(f.shape[0] for f in per_chunk_frames)),
            "chunk_lens": [int(f.shape[0]) for f in per_chunk_frames],
            "video": None,
            "sxs_video": None,
            "err": None,
        }
        try:
            streaming_rgb = np.concatenate(per_chunk_frames, axis=0)
            streaming_path = ep_dir / "long_video_streaming_decode.mp4"
            write_mp4(streaming_path, streaming_rgb, fps)
            streaming_decode_stats["video"] = str(streaming_path)
            print(f"[{ep_tag}] saved long_video_streaming_decode.mp4  "
                  f"shape={streaming_rgb.shape}  "
                  f"decode={'overlap' if streaming_overlap_decode else 'isolated'}")
            if anchor_rgb_for_sxs is not None:
                sxs_frames, _ = make_side_by_side(anchor_rgb_for_sxs, streaming_rgb)
                sxs_path = ep_dir / "long_video_streaming_decode_vs_anchor.mp4"
                write_mp4(sxs_path, sxs_frames, fps)
                streaming_decode_stats["sxs_video"] = str(sxs_path)
                print(f"[{ep_tag}] saved long_video_streaming_decode_vs_anchor.mp4")
        except Exception as e:
            streaming_decode_stats["err"] = repr(e)
            print(f"[{ep_tag}] long_video_streaming_decode.mp4 failed: {e!r}")

        # ----- Debug-only chunkwise-stitched mp4 kept for backwards
        # compatibility with existing artifact names. When overlap decode is
        # enabled this is equivalent to long_video_streaming_decode.mp4.
        if not skip_grounded_video:
            try:
                chunkwise_rgb = streaming_rgb if streaming_rgb is not None else np.concatenate(per_chunk_frames, axis=0)
                write_mp4(ep_dir / "long_video_chunkwise_debug.mp4", chunkwise_rgb, fps)
                print(f"[{ep_tag}] saved long_video_chunkwise_debug.mp4  "
                      f"shape={chunkwise_rgb.shape} (debug alias of streaming decode)")
            except Exception as e:
                print(f"[{ep_tag}] long_video_chunkwise_debug.mp4 failed: {e!r}")

        # ----- Save raw per-chunk latents for offline re-decode / inspection.
        try:
            if per_chunk_latents_cpu:
                np.savez_compressed(
                    ep_dir / "per_chunk_latents.npz",
                    **{f"chunk_{i:02d}": lat.to(torch.float32).numpy()
                       for i, lat in enumerate(per_chunk_latents_cpu)},
                )
        except Exception as e:
            print(f"[{ep_tag}] per_chunk_latents.npz save failed: {e!r}")

        # ----- Long side-by-side built against the production long_video.mp4
        # so the visual comparison still uses the unified-decode flavor.
        # If grounded video is skipped, the sxs will be built against the
        # AR-pass video below (deferred to after the AR pass).
        if not skip_grounded_video:
            try:
                if anchor_rgb_for_sxs is not None and long_rgb is not None:
                    sxs_frames, _ = make_side_by_side(anchor_rgb_for_sxs, long_rgb)
                    write_mp4(ep_dir / "long_video_vs_anchor.mp4", sxs_frames, fps)
                    print(f"[{ep_tag}] saved long_video_vs_anchor.mp4")
            except Exception as e:
                print(f"[{ep_tag}] long sxs failed: {e!r}")

        # ----- Opt-in: replay the same schedule in AUTOREGRESSIVE-VIDEO mode
        # (chunk N>0 receives latent_video=<chunk N-1's video_pred>) which
        # bypasses the per-chunk observation VAE re-encode at upstream
        # wan_flow_matching_action_tf.py line 1080-1100. The model autoregressively
        # extends its own predicted latent stream, giving a continuous
        # predicted-future video. Actions from this pass are DISCARDED --
        # the production action chunks come from the observation-grounded
        # pass above. Costs ~one extra full per-chunk inference loop per ep
        # (no model reload). Toggle with LONG_VIDEO_AUTOREGRESSIVE_DEBUG=1.
        if not ep_failed and _LONG_VIDEO_AUTOREGRESSIVE_DEBUG and per_chunk_latents_cpu:
            try:
                ar_long_rgb, ar_n_lat, ar_t_total = _run_autoregressive_debug_pass(
                    policy=policy,
                    ds=ds,
                    ep=ep,
                    ep_tag=ep_tag,
                    schedule=schedule,
                    prompt=prompt,
                )
                if ar_long_rgb is not None:
                    write_mp4(ep_dir / "long_video_autoregressive_debug.mp4",
                              ar_long_rgb, fps)
                    print(f"[{ep_tag}] saved long_video_autoregressive_debug.mp4  "
                          f"shape={ar_long_rgb.shape}  "
                          f"duration~{ar_long_rgb.shape[0] / fps:.1f}s  "
                          f"latent_cat=({ar_n_lat} t-frames)  "
                          f"second_pass_wall={ar_t_total:.1f}s")
                    # If we skipped the grounded video, emit the sxs against
                    # the AR-pass rgb so the user still gets an anchor-vs-prediction
                    # side-by-side artifact.
                    if skip_grounded_video and anchor_rgb_for_sxs is not None:
                        try:
                            sxs_frames, _ = make_side_by_side(anchor_rgb_for_sxs, ar_long_rgb)
                            write_mp4(ep_dir / "long_video_vs_anchor.mp4", sxs_frames, fps)
                            print(f"[{ep_tag}] saved long_video_vs_anchor.mp4 "
                                  f"(AR-pass video vs anchor)")
                        except Exception as e:
                            print(f"[{ep_tag}] AR sxs failed: {e!r}")
            except Exception as e:
                print(f"[{ep_tag}] autoregressive debug pass failed: {e!r}")
                traceback.print_exc()

        # Action chunks .npz + trajectory plot.
        try:
            if per_chunk_action:
                stack = np.stack(per_chunk_action, axis=0)  # (K, 24, 8)
                np.savez(
                    ep_dir / "action_chunks.npz",
                    action_stack=stack,
                    concat_trajectory=stack.reshape(-1, 8),
                    frame_indices=np.array([m["frames"] for m in per_chunk_meta], dtype=object),
                )
                try:
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt
                    traj = stack.reshape(-1, 8)
                    fig, ax = plt.subplots(figsize=(12, 5))
                    for d in range(8):
                        ax.plot(traj[:, d], label=f"a[{d}]", linewidth=0.8)
                    for ci in range(1, stack.shape[0]):
                        ax.axvline(ci * 24, color="k", alpha=0.2, linewidth=0.7)
                    ax.set_xlabel("horizon step")
                    ax.set_ylabel("action value")
                    ax.set_title(f"{ep_tag} concat trajectory  K={stack.shape[0]} chunks")
                    ax.legend(ncols=4, fontsize=8)
                    ax.grid(alpha=0.3)
                    fig.tight_layout()
                    fig.savefig(ep_dir / "trajectory.png", dpi=110)
                    plt.close(fig)
                except Exception as e:
                    print(f"[{ep_tag}] trajectory plot failed: {e!r}")
        except Exception as e:
            print(f"[{ep_tag}] action npz save failed: {e!r}")

        # Per-episode mem CSV.
        with open(ep_dir / "mem_per_chunk.csv", "w") as f:
            if per_chunk_meta:
                keys = list(per_chunk_meta[0].keys())
                f.write(",".join(keys) + "\n")
                for r in per_chunk_meta:
                    f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")

        ep_manifest = {
            "episode": ep,
            "result": "PARTIAL" if ep_failed else "PASS",
            "prompt": prompt,
            "episode_length": ep_len,
            "anchor_local_frame": anchor,
            "num_chunks_requested": num_chunks,
            "num_chunks_run": len(per_chunk_frames),
            "long_video_frames": int(sum(f.shape[0] for f in per_chunk_frames)),
            "long_video_duration_s": round(sum(f.shape[0] for f in per_chunk_frames) / fps, 2),
            "per_chunk": per_chunk_meta,
            "video_fps": fps,
            "fifo_action_mode": bool(fifo_action_mode),
            "defer_video_decode": bool(defer_video_decode),
            "fifo_action_events": fifo_action_events,
            "streaming_decode": streaming_decode_stats,
            "unified_decode": unified_decode_stats,
        }
        with open(ep_dir / "manifest.json", "w") as f:
            json.dump(ep_manifest, f, indent=2)
        per_ep_rows.append(ep_manifest)
        if not ep_failed:
            successes += 1
        print(f"[{ep_tag}] DONE  K_run={len(per_chunk_frames)}/{num_chunks}  "
              f"duration={ep_manifest['long_video_duration_s']}s  result={ep_manifest['result']}")

    # ---- Top-level long manifest -------------------------------------------
    banner("Long-form rollout summary")
    final_mem = cuda_mem_gib()
    long_manifest = {
        "stage": "stage5_long",
        "result": "PASS" if successes > 0 else "FAIL",
        "model_path": model_path,
        "embodiment_tag": embodiment,
        "dataset_dir": str(dataset_dir),
        "dataset_variant": getattr(ds, "variant", "unknown"),
        "episode_list_requested": episode_list,
        "episode_list_attempted": valid_episodes,
        "episodes_passed": [r["episode"] for r in per_ep_rows if r["result"] == "PASS"],
        "n_passed": successes,
        "n_failed_or_partial": sum(1 for r in per_ep_rows if r["result"] != "PASS"),
        "failures": failures,
        "anchor_local_frame": anchor_local,
        "num_chunks_per_episode": num_chunks,
        "video_fps": fps,
        "streaming_overlap_decode": {
            "enabled": bool(streaming_overlap_decode),
            "k_overlap": int(streaming_overlap_k if streaming_overlap_decode else 0),
        },
        "stage_b": {
            "enabled": _STAGE_B_ENABLED,
            "max_chunk_size": _STAGE_B_MAX_CHUNK_SIZE,
            "vae_cycle": os.environ.get("STAGE_B_VAE_CYCLE", "0"),
        },
        "stage_c": {
            "enabled": _STAGE_C_ENABLED,
            "encode_image_cache_hits": int(
                getattr(policy.trained_model.action_head,
                        "_stage_c_cache_hits", 0)
            ),
            "encode_image_cache_misses": int(
                getattr(policy.trained_model.action_head,
                        "_stage_c_cache_misses", 0)
            ),
        },
        "enable_flash_sdpa": os.environ.get("ENABLE_FLASH_SDPA", "0"),
        "flash_attn_backend": os.environ.get("FLASH_ATTN_BACKEND", "sdpa"),
        "flash_attention_triton_amd_enable": os.environ.get(
            "FLASH_ATTENTION_TRITON_AMD_ENABLE", "FALSE"
        ),
        "fifo_action_mode": bool(fifo_action_mode),
        "defer_video_decode": bool(defer_video_decode),
        "denoise_steps": os.environ.get("DENOISE_STEPS", "<default 16>"),
        "stage_f": {
            "decouple_inference_noise": bool(getattr(
                policy.trained_model.action_head.config,
                "decouple_inference_noise", False
            )),
            "video_inference_final_noise": float(getattr(
                policy.trained_model.action_head.config,
                "video_inference_final_noise", 0.8
            )),
        },
        "load_seconds": round(load_dt, 2),
        "total_seconds": round(time.perf_counter() - t_start, 2),
        "vram_peak_alloc_gib": round(final_mem["max_alloc_gib"], 3),
        "host_ram_peak_used_gib": round(max(r.get("used_gib", 0.0) for r in mem_trace), 2),
        "gpu": torch.cuda.get_device_name(0),
        "gpu_arch": torch.cuda.get_device_properties(0).gcnArchName,
        "torch_version": torch.__version__,
        "per_episode_summaries": per_ep_rows,
    }
    with open(output_dir / "long_manifest.json", "w") as f:
        json.dump(long_manifest, f, indent=2)
    print(f"[save] {output_dir/'long_manifest.json'}")

    with open(output_dir / "mem_trace.csv", "w") as f:
        keys = ["label", "ts", "used_gib", "free_gib", "alloc_gib", "max_alloc_gib", "reserved_gib"]
        f.write(",".join(keys) + "\n")
        for r in mem_trace:
            f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")

    banner("STAGE 5 LONG-FORM RESULT")
    print(f"  episodes attempted     : {len(valid_episodes)}")
    print(f"  episodes passed        : {successes}")
    print(f"  episodes partial/fail  : {long_manifest['n_failed_or_partial']}")
    print(f"  load seconds           : {load_dt:.1f}")
    print(f"  total wall seconds     : {time.perf_counter() - t_start:.1f}")
    print(f"  peak VRAM              : {final_mem['max_alloc_gib']:.2f} GiB")
    print(f"  output_dir             : {output_dir}")
    for r in per_ep_rows:
        print(f"    ep{r['episode']:>4}  K={r['num_chunks_run']:>2}/{num_chunks}  "
              f"dur={r['long_video_duration_s']:>5.1f}s  result={r['result']}")
    return 0 if successes > 0 else 9


if __name__ == "__main__":
    sys.exit(main())
