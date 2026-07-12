# vit_distill — MolmoAct2 vision-encoder distillation

Distill the frozen MolmoAct2 SigLIP2 ViT (~616 GFLOPs/crop) into a ~100x lighter
student that reproduces the concatenated `vit_layers` **seam** features
`[B, num_crops, 729, 2304]` feeding the frozen 2x2 attention pool + projector.

## Approach: plug into a mature KD framework (don't hand-roll)

Training is driven by **[torchdistill](https://github.com/yoshitomo-matsubara/torchdistill)**
(PyPI, PyTorch Ecosystem, MIT) — a config-driven KD engine with 26 SOTA methods,
DDP, forward-hook feature extraction, scheduling and checkpointing. We own only
thin adapters; the experiment is a YAML file.

- Student design provenance: **TinyViT** (microsoft/Cream) — MBConv-stem + light
  attention hybrid, the reference recipe for distilling a big ViT into a tiny one.
- Cross-arch alignment references: MobileCLIP, Clip4Retrofit, Align-KD.

## Layout

| file | role | torch? |
|------|------|--------|
| `distill/config.py` | teacher spec + student config (FLOP-budgeted) | no |
| `distill/flops.py` | analytic FLOP profiler (teacher + student) | no |
| `distill/student.py` | student encoder + `SeamStudent` crop-folding wrapper | yes |
| `distill/teacher.py` | frozen `SeamTeacher` (loads MolmoAct2 ViT) | yes |
| `distill/patchify.py` | frame -> normalized teacher-ready patches | yes |
| `distill/data.py` | LeRobot LIBERO frames + augmentation dataset | yes |
| `distill/losses.py` | seam cosine + normalized-MSE + fidelity metrics | yes |
| `distill/register.py` | registers teacher/student/loss with torchdistill | yes |
| `train.py` | torchdistill runner w/ seam-fidelity eval | yes |
| `configs/distill_hybrid.yaml` | the experiment | — |
| `tests/test_flops.py` | local torch-free validation | no |

## Local validation (no GPU/torch needed for these)

```bash
python tests/test_flops.py          # FLOP budget + config sanity
python -m distill.flops             # print teacher vs student compression
```

Default hybrid student: **5.4 GFLOPs = 114x** below the ~616 GFLOP teacher.

## Cluster run (needs the ROCm 7.2+ image; instinct gfx942 or instinct gfx950)

```bash
pip install torchdistill                      # baked into the image
torchrun --nproc_per_node=8 train.py \
    --config configs/distill_hybrid.yaml --run_log logs/run.log
```

## Portable image (instinct: docker images are node-local)

The built image is exported once to shared NFS as a tarball and loaded on demand,
so a job can run on **any** node (not just the build node):

```bash
# produced by scratch/build_save_m1.sbatch:
#   /shared/$USER/vit_distill/images/molmoact2-vitdistill-rocm950.tar.gz  (~11 GB)
IMG=molmoact2-vitdistill:rocm950 bash scratch/ensure_image.sh   # docker-load if absent
```

Build steps that hit the network (apt/pip) require `docker build --network=host`
(the default docker0 bridge is broken on some workers). Model/dataset loads run with
`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` once the HF cache under `/shared/.../hf_cache`
is warm (unauthenticated HF metadata calls otherwise stall on rate limits).

## M1 — teacher validated on gfx950 (MI355X), `artifacts/vit_distill/m1_teacher.json`

All checks PASS on a real forward pass of the frozen `allenai/MolmoAct2-LIBERO` ViT:

1. Processor patchify → `[1, 729, 588]` float32; raw patches in `[-1, 1]`.
2. `vit_layers` resolve to `[24, 18]` (i.e. `-3,-9` on the 27-layer ViT → 25 resblocks run).
3. `SeamTeacher` seam `[B, 1, 729, 2304]` bf16, finite (mean≈0, std≈3.07), **deterministic**
   across repeated forwards (max-abs-diff `0.0`).
4. Normalization parity `0.0`: `already_normalized` True/False paths agree, and
   `normalize_patches` reproduces `MolmoAct2VisionBackbone.forward` scaling exactly.
5. **Finding (bf16 sensitivity):** normalizing on GPU vs CPU differs by one bf16 ULP
   (`3.05e-5`) at the input, which the deep bf16 ViT amplifies to `~1.0` max in the seam.
   The pipeline avoids this by normalizing **once** (CPU float32) so teacher and student
   consume the *identical* tensor; targets are harvested online for the same reason.

torchdistill stores the model root output under path `'.'` (the seam loss reads it).

## Data pipeline — direct-parquet reader + fast patchify

The MolmoAct2-LIBERO dataset stores each camera frame as PNG bytes inside its
`data/*.parquet` files (no separate videos). `distill/data.py` reads them straight
with pyarrow + PIL, bypassing LeRobot's episode/video index (which also trips a
lerobot↔huggingface_hub version bug; see `distill/_lerobot_compat.py`).

**Patchify fast-path (key optimization):** patchifying via the full multimodal
processor (`processor(images=..., text=...)`) cost ~24 fps because it also runs
text/video tokenization. Calling `processor.image_processor.preprocess(...)`
directly yields **bit-identical** `pixel_values` (`max|Δ| = 0.0`, verified by
`scratch/patchify_speed.py`) at **407 fps single-thread** (~17×). Training is now
GPU-bound (`data ≈ 0.0002 s/step`).

## M2 — single-GPU smoke (`configs/distill_smoke.yaml`)

Proved the loop end-to-end: loss `1.00 → 0.56`, val seam cosine `0.56 → 0.64`,
checkpoint saves on each new best. PASS.

## M3 — full distillation (`configs/distill_full.yaml`, `scratch/m3_full.sbatch`)

4-GPU DDP over the whole LIBERO frame set (both cameras, ~518k views/epoch) with
heavy augmentation, AdamW (lr 1e-3, wd 0.05), cosine anneal, 10 epochs.

- **Training time 3:30:06** (~0.28 s/step, 128 frames/step, ~3.8 GB/GPU).
- Val seam cosine climbed `0.855 → 0.867 → 0.873 → 0.877 → 0.879 → 0.880 →
  0.882 → 0.883 → 0.884 → 0.885` (epoch 0→9); **best 0.88504**, rel-L2 0.556.
- **Test (held-out): seam cosine 0.885, rel-L2 0.556.** Checkpoint
  `resource/ckpt/vit_distill/hybrid_full.pt` (~33 MB, ~8M params).
- *Note:* used 4 GPUs (not 8) because the cluster had no node with 8 free GPUs;
  VRAM headroom is large, so this was not a quality constraint.

## M4 — fidelity eval (`scratch/m4_eval.py`, `scratch/m4_eval.sbatch`)

Held-out cosine/rel-L2 distribution + a two-column **GT (teacher) vs prediction
(student)** seam panel (shared teacher-fit PCA→RGB, rule 2.a). Outputs
`artifacts/vit_distill/m4_metrics.json` and `m4_seam_fidelity.png`.

Over 1024 held-out frames:

| metric | mean | std | p05 | p50 | p95 | min | max |
|--------|------|-----|-----|-----|-----|-----|-----|
| cosine | **0.885** | 0.018 | 0.849 | 0.888 | 0.908 | 0.815 | 0.915 |
| rel-L2 | 0.556 | 0.023 | 0.518 | 0.554 | 0.595 | 0.501 | 0.620 |

The panel shows the student reproduces the teacher's seam spatial structure and
PCA colour layout across scenes (slightly smoother high-frequency detail, as
expected for a ~100× smaller model). The tight cosine spread (p05 0.849, p95
0.908) indicates consistent fidelity rather than a few easy frames carrying the
mean.
