# MolmoAct2 &times; LIBERO-plus &mdash; curriculum-LoRA training + full evaluation

Reproduces the MolmoAct2 curriculum-LoRA fine-tune on the **LIBERO-plus** robustness benchmark and
its full closed-loop evaluation, on AMD ROCm GPUs. Five curriculum stages grow the LoRA rank
`16 -> 32 -> 64 -> 128 -> 256` (`s1..s5`), each warm-started from the previous stage merged back
into the base. `s5` (rank-256) is the best checkpoint.

This branch is the reproducible baseline for the follow-on **attention-feedback pruning** work on
the LIBERO-plus checkpoint.

## Results (this pipeline)

Success rate (%). `s1/s4/s5` are the full 10,030-task benchmark; `s2/s3` are the representative
1,673-task stratified subset (the full-vs-subset gap on `s4/s5` is < 1 pt).

| Stage | LoRA rank | Basis | Overall | spatial | object | goal | libero_10 |
|-------|-----------|-------|---------|---------|--------|------|-----------|
| s1 | 16  | full 10,030   | 67.2 | 84.0 | 79.8 | 74.6 | 30.9 |
| s2 | 32  | subset 1,673  | 68.6 | 84.3 | 79.8 | 78.2 | 32.4 |
| s3 | 64  | subset 1,673  | 68.9 | 86.3 | 77.6 | 80.3 | 31.9 |
| s4 | 128 | full 10,030   | 68.4 | 86.0 | 80.2 | 77.1 | 30.8 |
| s5 | 256 | full 10,030   | **69.6** | 86.5 | 81.4 | 79.5 | 31.7 |

Full per-perturbation (7 dims) and per-difficulty (L1&ndash;L5) breakdowns are in
[`results/rank_progression_full_breakdown.json`](results/rank_progression_full_breakdown.json).
The rank curve is shallow (+2.4 pt for 16&times; the rank); `libero_10` long-horizon (~31%) and the
`Robot Initial States` perturbation (~53%) are the dominant bottlenecks and do not improve with rank.

## Layout

```
docker/   Dockerfile.train  (ROCm + lerobot[molmoact2] + accelerate)
          Dockerfile.eval   (adds the LIBERO-plus CPU sim + OSMesa render)
train/    stage_data.py  train.sh  train.sbatch  merge_curriculum.py  merge.sbatch
policy/   filter_config.py  rewrap_policy.py  gen_processors.py  build_policy.sbatch
eval/     gen_full_plan.py  gen_subset_plan.py  eval_shard_ids.sh
          sweep_node.sh  sweep.sbatch  pick_node.sh  aggregate.py
results/  rank_progression_full_breakdown.json
```

## Prerequisites

- An AMD ROCm GPU host with Docker (`--device=/dev/kfd --device=/dev/dri`). Training used a single
  8-GPU node; a single GPU also works with smaller `BS`/`NPROC`.
- ~50 GB free for the base checkpoint (~21 GB) + dataset (~23 GB) + HF cache.
- HuggingFace access to the public repos `allenai/MolmoAct2-LIBERO` and `Sylvest/libero_plus_lerobot`.

All large binaries (base checkpoint, dataset, 6.4 GB perturbation assets) are downloaded by the
scripts / Dockerfiles at build time &mdash; nothing is vendored here.

Set one environment root used throughout:

```bash
export WORK=$HOME/molmoact2           # project root (checkpoints, cache, outputs)
export HF_ROOT=$WORK/hf_cache
```

## 1. Build the images

```bash
cd docker
docker build -t molmoact2-lerobot-train:rocm950     -f Dockerfile.train .
docker build -t molmoact2-libero-plus-eval:rocm950  -f Dockerfile.eval  .   # FROM the train image
```

## 2. Download the base checkpoint + dataset + assets

```bash
HF_ROOT=$WORK/hf_cache python train/stage_data.py
```

## 3. Train the curriculum (LoRA rank 16 -> 256)

Each stage LoRA-fine-tunes the VLM on LIBERO-plus, then `merge_curriculum.py` folds the adapter
into a plain-base checkpoint so the next stage warm-starts at a higher rank. Base recipe: bf16,
`chunk_size=10`, gradient checkpointing on (set `GRAD_CKPT=false` on GPUs that lack it), one epoch
over ~2.24 M frames (`steps = ceil(2238036 / (NPROC*BS))`).

**Single machine** (run stages sequentially, merging between):

```bash
# stage s1 (rank 16), warm-started from the released checkpoint
WORK=$WORK RUN=lp_curric_s1 LRANK=16 CKPT=allenai/MolmoAct2-LIBERO \
  STEPS=34969 BS=8 NPROC=8 bash train/train.sh

# merge s1 -> plain base, then train s2 (rank 32) from it, and so on for s3/s4/s5
SRC=$WORK/outputs/lp_curric_s1/checkpoints/last/pretrained_model \
DST=$WORK/outputs/lp_curric_merged/s1 \
  docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
    -e HOME=/cache -e HF_HOME=/cache -e SRC=/outputs/lp_curric_s1/checkpoints/last/pretrained_model \
    -e DST=/outputs/lp_curric_merged/s1 -v $WORK/hf_cache:/cache -v $WORK/outputs:/outputs \
    -v $PWD/train:/work molmoact2-lerobot-train:rocm950 python3 /work/merge_curriculum.py
```

**Cluster (SLURM)** &mdash; self-chaining fixed-length segments (fits a per-job walltime cap),
auto-merge + auto-advance to the next stage:

```bash
sbatch -p "$GPU_PARTITION" \
  --export=ALL,WORK=$WORK,STAGE=1,LRANK=16,STEPS=34969,BS=8,GPU_PARTITION=$GPU_PARTITION \
  train/train.sbatch
```

`train.sbatch` resubmits itself until the stage target is reached, then submits `merge.sbatch`,
which merges and launches the next stage. The final stage (`s5`) stops the chain.

## 4. Build lerobot policy dirs from the merged checkpoints

`lerobot-eval` needs a lerobot-format policy dir. Copy any trained stage's `config.json` to
`$WORK/eval_ckpts/_lerobot_cfg_src.json` once, then:

```bash
WORK=$WORK STAGES="s1 s2 s3 s4 s5" sbatch policy/build_policy.sbatch
# per stage: filter_config.py -> rewrap_policy.py -> gen_processors.py (norm_tag=libero)
```

## 5. Evaluate (closed loop)

Generate a plan (full 10,030 or the 1,673 stratified subset), then run the sweep. 4 concurrent
OSMesa renders per node is the CPU-render contention sweet spot; each shard writes `eval_info.json`
for resume-by-shard.

```bash
mkdir -p $WORK/plans
python eval/gen_full_plan.py   node 4 55            $WORK/plans/plan_full_s5.tsv     # full 10,030
python eval/gen_subset_plan.py node 1673 4 8        $WORK/plans/plan_subset_s5.tsv   # subset 1,673
```

**Single machine** (one GPU, sequential shards):

```bash
KIND=full WORK=$WORK IMG=molmoact2-libero-plus-eval:rocm950 \
  bash eval/sweep_node.sh $WORK/plans/plan_full_s5.tsv localhost s5
```

**Cluster (SLURM)** &mdash; self-chaining segments with self-healing node placement:

```bash
sbatch -p "$EVAL_PARTITION" \
  --export=ALL,WORK=$WORK,STAGE=s5,KIND=full,EVAL_PARTITION=$EVAL_PARTITION \
  eval/sweep.sbatch
```

## 6. Aggregate

```bash
python eval/aggregate.py \
  --base $WORK/libero_plus_eval \
  --classmap /opt/libero-plus/libero/libero/benchmark/task_classification.json \
  --stages s1:rank-16 s2:rank-32 s3:rank-64 s4:rank-128 s5:rank-256 \
  --kind full
```

Produces overall + per-suite + per-perturbation + per-difficulty success rates (the shape of
`results/rank_progression_full_breakdown.json`).

## Notes

- **bf16** is the model's intended inference/train precision; merged checkpoints are saved bf16.
- **Best checkpoint**: stage `s5` (rank-256), the merged plain-base bf16 model (~11 GB), produced by
  step 3. It is not vendored here &mdash; reproduce it with the pipeline above.
- Compute-only CDNA GPUs have no graphics engine, so the sim renders headless on CPU via OSMesa.
- Partitions/nodes are passed via `GPU_PARTITION` / `EVAL_PARTITION`; the eval sweep is
  node-agnostic (matches only the GPU column of the plan).
