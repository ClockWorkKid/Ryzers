# Upstream pins

All upstream artifacts this package depends on, with exact revisions.

## DreamZero source

- Repo: <https://github.com/dreamzero0/dreamzero>
- License: Apache 2.0
- Pinned commit: `ab790c198fbce33503358efbbd4187ce9a89adf3` (`ab790c1`, "Update architecture
  differences in WAN22_BACKBONE.md", authored 2026-04-19)
- The Dockerfile reproduces this exactly via `--build-arg DREAMZERO_REF=…` (default is the full
  SHA); the clone is placed at `/opt/dreamzero`, first on `PYTHONPATH`, and never edited.

## Hugging Face models / data

| Artifact | HF repo | Approx. size | Revision |
|---|---|---:|---|
| DreamZero-DROID (14B checkpoint) | `GEAR-Dreams/DreamZero-DROID` | ~28 GB (bf16) | `main` at download time |
| Wan2.1 base (I2V, 14B 480P) — VAE/CLIP + UMT5 t5 encoder | `Wan-AI/Wan2.1-I2V-14B-480P` | ~40 GB | `main` at download time |
| UMT5-XXL text encoder tokenizer/config | `google/umt5-xxl` | ~11 GB | `main` at download time |
| DROID eval data (per-episode LeRobot) | `GEAR-Dreams/DreamZero-DROID-Data` | large (fetch a few episodes) | `main` at download time |

Fetched by `scripts/download_checkpoints.sh`. Not revision-pinned by default; pin here if
reproducing bit-exact.

## Docker base image

- `init_image` is **omitted** in `config.yaml`: ryzers defaults to the ROCm base
  (`rocm/pytorch`, torch 2.10.0) that ships the gfx1151 ROCm torch stack.
- Validated on this base under both **rocm7.2.2** and **rocm7.14** (torch `2.10.0+rocm7.14.0`
  on the 7.14 build). The Dockerfile is otherwise base-agnostic — no CUDA/flash-attn build; the
  overlay SDPA shim + aotriton provide attention on ROCm.
- The only version-sensitive runtime knob is the allocator: `PYTORCH_HIP_ALLOC_CONF=expandable_segments:False`
  (required on 7.14; safe on 7.2.2). See `RUNTIME_OPTIMIZATION.md`.
