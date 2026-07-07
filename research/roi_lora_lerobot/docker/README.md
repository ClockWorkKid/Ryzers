# MolmoAct2 + LeRobot LoRA training images (ROCm)

Two Dockerfiles that codify the *same* validated training environment on two AMD
targets. Both install the LeRobot `molmoact2` policy (with native PEFT LoRA) on
top of a ROCm PyTorch base, and both were smoke-tested end-to-end (10-step
LoRA-VLM fine-tune on LIBERO, `rc=0`).

| File | Target | Base image | GPU arch |
| --- | --- | --- | --- |
| `Dockerfile.instinct-gfx942` | AMD Instinct MI300X | `rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.10.0` | gfx942 |
| `Dockerfile.strix-gfx1151` | AMD Strix Halo APU (dev box) | `ryzer_env:latest` (Ryzers ROCm 7.2 / torch 2.10) | gfx1151 |

## Why these choices
- **Pinned LeRobot commit `052d329`** (v0.5.2 dev) — the release on PyPI is only
  0.5.1 and does not include the `molmoact2` policy, so we clone + editable-install.
- **torch trio pinned by public version** (local `+rocm...` segment stripped) so
  `pip` keeps the base ROCm wheels and never swaps in CUDA builds.
- **No `torchcodec`** — it hard-pins torch and does not load on ROCm. We install
  the rest of LeRobot's `[dataset]` deps (`datasets`, `pandas`, `pyarrow`) and use
  the **pyav** video backend (`--dataset.video_backend=pyav`).
- **ffmpeg dev headers** are apt-installed because PyAV has no matching manylinux
  wheel here and builds from source.
- `TORCH_BLAS_PREFER_HIPBLASLT=0` routes the small action-expert GEMM fallbacks.
  Strix also sets `HSA_OVERRIDE_GFX_VERSION=11.5.1` and AOTriton experimental attn.

## Build
```bash
# Instinct / MI300X
docker build -t molmoact2-lerobot-train:rocm942 -f Dockerfile.instinct-gfx942 .

# Strix Halo (build ryzer_env first via the ryzers package system)
docker build -t molmoact2-lerobot-train:rocm    -f Dockerfile.strix-gfx1151 .
```

## Reference LoRA-VLM fine-tune command
Run inside the container (mount an HF cache at `$HF_HOME` and an output dir):
```bash
accelerate launch --num_processes=1 --mixed_precision=bf16 \
  -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=allenai/MolmoAct2-LIBERO-Dataset \
  --dataset.revision=main --dataset.video_backend=pyav \
  --policy.type=molmoact2 --policy.checkpoint_path=allenai/MolmoAct2-LIBERO \
  --policy.push_to_hub=false \
  --policy.enable_lora_vlm=true --policy.action_mode=both \
  --policy.chunk_size=10 --policy.n_action_steps=10 \
  --policy.model_dtype=bfloat16 --policy.num_flow_timesteps=8 \
  --policy.gradient_checkpointing=true \
  --batch_size=2 --steps=10 --save_freq=1000 --output_dir=/outputs/run1
```
Notes: run the container as the host user (`--user`) with `-e USER=<name>` set so
`getpass.getuser()` works without an `/etc/passwd` entry; expose GPUs with
`--device=/dev/kfd --device=/dev/dri --group-add video --group-add render`.
