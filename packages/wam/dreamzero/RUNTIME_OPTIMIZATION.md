# DreamZero-DROID on Strix Halo (gfx1151) - runtime notes

WAM-direct philosophy: upstream `/opt/dreamzero` stays pristine; every AMD-specific fix is a
runtime monkey-patch in `overlay/` so the pinned commit and the paper's numerics are preserved.
This doc records the two things that make the 14B model run correctly on a unified-memory iGPU:
the **memory strategy** and the **open-loop looping fix**.

## Hardware

AMD Ryzen AI Max+ 395 (Strix Halo), `gfx1151`, single ROCm base (torch 2.10.0). CPU and GPU
share one ~94 GiB unified pool; the GPU's usable GTT carveout is ~47 GiB. There is no discrete
VRAM - every allocation competes with host RAM, so peak transient allocation matters more than
steady-state resident size.

## Attention: SDPA shim, no flash-attn build

`flash_attn.*` is CUDA-only upstream. `overlay/python/flash_attn` is a thin re-export shim that
satisfies the import surface (`flash_attn_func`, `flash_attn_varlen_func`, `bert_padding`, …) and
routes to torch SDPA. With `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` the SDPA path uses
aotriton fused attention on gfx1151. This removes the ROCm flash-attention source build entirely
(`FLASH_ATTN_BACKEND=sdpa`, `ATTENTION_BACKEND=torch`). Smoke-tested by `test.py`
(`flash_attn 0.0.0+sdpa-shim`).

## Memory strategy (why 14B fits in ~42-43 GiB)

The naive `.to("cuda")` load and per-chunk VAE decode blow past the GTT pool. Fixes, all applied
at runtime by `validate_stage3_path_b.py::apply_amd_patches` + `overlay/tests/_stage_b_patches.py`:

| Fix | Effect |
|---|---|
| **Allocator: `expandable_segments:False`** | On **rocm7.14**, `True` silently caps torch at ~19.5 GiB of the GTT pool (load OOMs); `False` reaches the full ~47 GiB. On 7.2.2 either works; we standardize on `False`. |
| **Tiled VAE encode + decode** (`vae.encode/decode -> tiled=True`) | Per-tile conv3d activation drops from ~4.35 GiB to ~0.1-0.5 GiB - the single biggest decode-time saver. |
| **Text-encoder CPU offload** | UMT5-XXL (~11 GiB) is evicted to CPU after the prompt encode; frees headroom for the DiT forward. Re-materialized on demand. |
| **Forced-CUDA `prepare_input`** | With the text encoder on CPU, upstream infers device from `next(parameters())` and ripples CPU tensors into a CUDA image encoder → device-mismatch. The patch pins inputs to CUDA. |
| **`max_chunk_size` 4 → 2** | Bounds the video-latent working set per forward. |

Compile paths (`DREAMZERO_COMPILE_*`, dynamo) are disabled on ROCm.

## The open-loop "looping" artifact

In open-loop rollout the predicted future-video looked like it *repeated / redid the task*.
Root cause was three compounding effects - **none is a model bug**:

1. **Playback rate.** The model's predicted video is native **5 fps** (upstream saves at fps=5).
   Rendering it at the DROID dataset's 15 fps plays it ~3.2× too fast, so one reach reads as
   several. → `VIDEO_FPS=5`.
2. **Per-chunk VAE boundary spike.** Decoding each chunk's latent in isolation drops temporal
   context at the seam and produces a visible discontinuity. → **rolling-overlap decode**: prepend
   the previous chunk's latent tail, decode, emit only the new RGB frames
   (`STREAMING_OVERLAP_DECODE=1`, `STREAMING_OVERLAP_K=1`; ~49% seam-spike reduction).
3. **Image-feature reuse at local-attention resets.** Re-using chunk-0 image features across a
   reset injects a semantic jump. → `STAGE_C=0`.

For honest long-horizon visualization beyond the grounded schedule, `LONG_VIDEO_AUTOREGRESSIVE_DEBUG=1`
replays the schedule feeding each chunk's `video_pred` as the next chunk's latent (the model
extends its own predicted stream) - `long_video_autoregressive_debug.mp4`.

### Evidence

With the fix on, per-chunk decoded lengths are `[9, 8, 12, 8]` (overlap-continuous) rather than
the isolated-decode `[9, 5, 9, 5]` truncation pattern, and the unified re-decode of the
concatenated latents matches the streamed frame count (10 latent → 37 RGB). Side-by-side
`anchor | prediction` strips show a single coherent, monotonically-progressing rollout.

## Validated run (rocm7.14)

`GEAR-Dreams/DreamZero-DROID`, episode 0 (`Pick up the blue ring …`), K=4, `DENOISE_STEPS=2`,
looping fix on:

| Metric | Value |
|---|---|
| Result | PASS (1/1 episodes) |
| torch | 2.10.0+rocm7.14.0, `AMD Radeon 8060S` / gfx1151 |
| VRAM peak (alloc) | **41.9 GiB** (per-chunk max 43.1 GiB; unified decode 36.7 GiB) |
| Host RAM peak | 46.8 GiB |
| Load | 282.8 s | 
| Total | 494.3 s |
| Per-chunk infer | 17-32 s | 
| Decode mode | `overlap` (all chunks) |

### DROID data layout note

The released eval data (`GEAR-Dreams/DreamZero-DROID-Data`) uses a **per-episode** LeRobot layout
(`data/chunk-000/episode_NNNNNN.parquet` + `videos/chunk-000/<cam>/episode_NNNNNN.mp4`,
`meta/episodes.jsonl` + `meta/modality.json`) rather than the droid_1.0.1 sharded layout. The
overlay `DroidDataset` gained a `droid_gear` variant that auto-detects this, slices the 14-dim
`observation.state` → 8-dim `[joint(7)+gripper(1)]` via `modality.json`, and decodes frames by
index (one episode per mp4). State/action semantics are read from `modality.json`, not hardcoded.
