# MolmoAct2 distillation (research)

Two research tracks that compress MolmoAct2 for efficient / edge deployment while
retaining LIBERO closed-loop task success. Training runs on AMD Instinct GPUs
(MI300-class `gfx942`, MI325-class `gfx950`) via Apptainer + the eval image.

- **`vit/`** — SigLIP2 ViT encoder distillation into **~100x-lighter** students
  (initial hybrid distill + three finalized variants A/B/C), each with a LoRA
  finetune on top of the frozen distilled encoder.
- **`llm/`** — thin-twin LLM distillation, then a **joint** distilled-ViT +
  distilled-LLM finetune on a LIBERO+DROID mix, then a LoRA adaptation stage
  (Run D). Uses per-layer KV adapters (208 -> 1024) into the frozen flow-matching
  action expert.

## ViT distillation — finalized results (100-episode LIBERO, N-way)
25 episodes/suite x 4 suites = 100 episodes/arm, identical seeds. All three
~100x-compressed variants retain **>=98%** of the teacher and beat the plain
finetuned-hybrid baseline (95%).

| arm | Spatial | Object | Goal | Long | Overall | params | GFLOPs/crop | comp |
|-----|--------:|-------:|-----:|-----:|--------:|-------:|------------:|-----:|
| teacher (SigLIP2 ViT) | 100 | 100 | 100 | 100 | **100.0** | - | 617.15 | 1.0x |
| distill-only hybrid | 100 | 96 | 92 | 88 | 94.0 | 2.79M | 5.40 | 114x |
| finetuned hybrid | 100 | 100 | 92 | 88 | 95.0 | 2.79M | 5.40 | 114x |
| **A - siglip_nano** (attn-only) | 100 | 100 | 96 | 96 | **98.0** | 2.73M | 5.74 | 107x |
| **B - hybrid_droid** (+DROID real) | 100 | 96 | 100 | 100 | **99.0** | 2.79M | 5.40 | 114x |
| **C - cnn_fpga** (pure-CNN) | 100 | 96 | 96 | 100 | **98.0** | 3.83M | 5.14 | 120x |

**B (hybrid_droid) is the recommended default at 99%** — mixing real DROID frames
into the seam-retention distillation is the only recipe that fully recovers the
hardest long-horizon suite. Raw numbers in `vit/results/compare_nway.csv`; full
run map in `vit/RESULTS.md`.

## Run D (LLM + joint) — status
Thin-twin student LLM (36 layers, hidden 208) replaces the teacher text
transformer; its per-layer KV feeds the frozen action expert via learned
adapters. Pipeline: Stage-1 LLM distill -> joint distilled-ViT+LLM finetune on
LIBERO+DROID (flow-matching + multi-level LLM KD + DROID seam-retention) -> LoRA
adaptation -> closed-loop eval. See `llm/README.md`.

## Reproduce (AMD Instinct)
1. Build the image from `vit/docker/Dockerfile.instinct-gfx942` (MI300-class) or
   `Dockerfile.instinct-gfx950` (MI325-class). The Dockerfile fetches all
   open-source assets itself; no weights are vendored here.
2. **ViT track**: `vit/scratch/closedloop/submit_pipeline.sh` chains distill ->
   LoRA finetune (12k) -> 100-ep LIBERO eval -> N-way aggregate. Per-variant
   kwargs are keyed by `NAME` (siglip_nano / hybrid_droid / cnn_fpga).
3. **LLM / Run D track**: `llm/scratch/{llmd_stage1,llmd_joint,llmd_lora,llmd_eval}.sbatch`
   (chain with SLURM `afterany`).

Paths use `$USER` / SLURM `%u`; set your own cluster partitions and a
`/shared_nobackup/$USER` workspace. Model weights, container images, and large
generated videos are not committed (built/pulled at runtime).
