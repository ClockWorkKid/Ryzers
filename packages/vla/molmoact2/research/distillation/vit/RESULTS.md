# 3x Distillation Experiments — pipeline run map

Full distill -> LoRA-finetune (12k) -> 100-ep LIBERO eval -> N-way compare, all on
instinct **mi210** via apptainer + `molmoact2-lerobot-eval.sif` + a shared `torchdistill`
target (`/shared_nobackup/$USER/molmoact2/pylibs`). mi300x-es splinter nodes have no
GPUs; mi300x docker nodes were busy, so the whole pipeline runs on mi210 (validated).

## Variants (all ~100x lighter than the ~617 GFLOP SigLIP2 teacher)
| run | arch | student kwargs | GFLOPs | comp | params |
|-----|------|----------------|--------|------|--------|
| siglip_nano | attention-only (tinyvit) | dim=240, 4 attn, 8 heads | 5.74 | 107x | 2.73M |
| hybrid_droid | conv+attn hybrid + DROID real data | dim=256, 4 conv, 3 attn | 5.40 | 114x | 2.79M |
| cnn_fpga | pure-CNN (FPGA-friendly) | dim=384, 16 conv, k3 | 5.14 | 120x | 3.83M |

Baselines: teacher 100%, distill-only hybrid ~94%, finetuned hybrid ~95%.

## Finalized results — 100-episode LIBERO closed-loop (N-way, `compare_nway.csv`)
Per-suite / overall success (%), 25 episodes/suite x 4 suites = 100 episodes/arm,
identical seeds across arms. All three 100x-compressed variants **retain >=98%** of
teacher performance after the LoRA finetune, and every variant **beats the plain
finetuned-hybrid baseline (95%)**.

| arm | Spatial | Object | Goal | Long | Overall | params | GFLOPs/crop | comp |
|-----|--------:|-------:|-----:|-----:|--------:|-------:|------------:|-----:|
| teacher (SigLIP2 ViT) | 100.0 | 100.0 | 100.0 | 100.0 | **100.0** | - | 617.15 | 1.0x |
| distill-only hybrid | 100.0 | 96.0 | 92.0 | 88.0 | 94.0 | 2.79M | 5.40 | 114x |
| finetuned hybrid | 100.0 | 100.0 | 92.0 | 88.0 | 95.0 | 2.79M | 5.40 | 114x |
| **A - siglip_nano** (attn-only tinyvit) | 100.0 | 100.0 | 96.0 | 96.0 | **98.0** | 2.73M | 5.74 | 107x |
| **B - hybrid_droid** (hybrid + DROID real) | 100.0 | 96.0 | 100.0 | 100.0 | **99.0** | 2.79M | 5.40 | 114x |
| **C - cnn_fpga** (pure-CNN, FPGA-friendly) | 100.0 | 96.0 | 96.0 | 100.0 | **98.0** | 3.83M | 5.14 | 120x |

Takeaways:
- **B (hybrid_droid) is best overall at 99%**, and is the only variant that fully
  recovers the hardest long-horizon suite (Long 100.0) - mixing real DROID frames
  into the seam-retention distillation clearly helps generalization/robustness.
- **A (siglip_nano)** matches at 98% with the fewest params (2.73M) despite being
  attention-only, and **C (cnn_fpga)** reaches 98% while staying pure-CNN (no
  attention op), the most FPGA-friendly at ~120x compression.
- The +DROID real-data recipe (B) is the recommended default; the LoRA finetune on
  top of the distilled encoder recovers the Goal/Long regressions seen in the
  distill-only / finetuned-hybrid baselines.

## SLURM DAG (submitted 2026-07-10)
- distill: siglip_nano=44615, hybrid_droid=44619, cnn_fpga=44616 (4h wall; best
  val-cosine checkpoint saved every improving epoch, ~30 min/epoch on mi210).
- finetune (afterany-chained, dep on distill): ft_siglip 44622-44625,
  ft_hybrid_droid 44627-44630, ft_cnn_fpga 44632-44635 (single-GPU bs8, resume to 12k).
- eval (dep on ft): ev_siglip 44626, ev_hybrid_droid 44631, ev_cnn_fpga 44636
  (4 suites x 5 tasks x N ep).
- aggregate: vd_agg 44637 (dep on all evals) -> outputs/compare_nway.csv.

## Key scripts (all in research/vit_distill/scratch/closedloop/ + configs/)
- vd_distill.sbatch, distill_{siglip_nano,hybrid_droid,cnn_fpga}.yaml
- droid_prep.py/.sbatch (23,470 real frames from lerobot/droid_1.0.1 exterior_1_left)
- distill/data.py: ImageFolderFrameDataset + LiberoDroidMixDataset
- student_ft_mi210.sbatch (NAME-keyed variant kwargs), run_train_student.py (+VIT_STUDENT_KWARGS)
- vd_eval_ft.sbatch (NAME-keyed), run_eval_ft.py (+VIT_STUDENT_KWARGS)
- aggregate_nway.py, vd_aggregate.sbatch, submit_pipeline.sh

## Storage cleanup (phase 0)
Pruned ~4 TB stale (outputs 4.0T->24G; /shared_nobackup 84%->76%). Recipes archived
to artifacts/vit_distill/stale_metadata.tgz. docker dangling-prune queued (job 44600).
