# Reproduce: MolmoAct2 LLM-backbone distillation → `qwen06w` (47.5% closed-loop)

End-to-end backup + reimplementation of the winning route. Every stage runs inside the eval SIF
via `apptainer exec` (see `docker/README.md`). See `DISTILLATION_SUCCESS_REPORT.md` for the method
and results, and `HYBRID_DISTILL_REPORT.md` for the full experiment record (incl. negative controls).

## 0. Prerequisites

- Build the eval SIF: `docker/build_eval_sif.sh` (see `docker/README.md`).
- Lay out a workspace root `WORK` with:
  - `WORK/images/molmoact2-lerobot-eval.sif`
  - `WORK/llm_distill/` ← this folder's `*.py` (bound to `/work`)
  - `WORK/hf_cache/` (HF cache; MolmoAct2 teacher weights pulled here)
  - `WORK/outputs/` (bound to `/outputs`; all checkpoints/rollouts land here)
  - `WORK/roi_overlay/` ← copy of `../roi_lora_lerobot/docker/roi_overlay/*`
- Every sbatch is path-portable: `export WORK=/your/workspace` (and optionally `OV`, `SIF`, `CODE`,
  `OUTPUTS`) before `sbatch`, and set `#SBATCH -p` to your GPU partition. Runs used a single MI210.
- `mkdir -p logs` next to the sbatch scripts (Slurm `--output` writes there).

## 1. Component validation (do this first — .cursorrules rule 2)

```bash
export WORK=/your/workspace
sbatch slurm/validate_student.sbatch   # StudentTextModel KV contract vs teacher on a real LIBERO batch
sbatch slurm/validate_joint.sbatch     # end-to-end functional path: attach student -> flow loss -> backward
sbatch slurm/dagger_validate.sbatch    # teacher relabel sanity (label vs demo MSE, flow self-consistency)
```

## 2. Stage 0 — warm-start `qwen06w` from Qwen3-0.6B

```bash
PRESET=qwen06w sbatch slurm/warmstart.sbatch
# -> /outputs/llm_distill/warmstart/qwen06w_init.pt
```

## 3. Stage 1 — all-layer teacher-KV anchored distillation (the enabling lever)

```bash
ANCHOR_W=1.0 KEEP_TEACHER=1 STEPS=6000 \
  OUTDIR=/outputs/llm_distill/runs/qwen06w_anchor \
  sbatch slurm/train_joint.sbatch
# reference "frozen-AE + KV-anchor" student ~30% closed-loop
```

## 4. Stage 2 — LLM + action-expert co-adaptation → LoRA fine-tune (the flexible `expd` base)

```bash
# Phase 1: co-adapt student LLM + full action expert under the KV anchor
ANCHOR_W=1.0 KEEP_TEACHER=1 STEPS=12000 \
  EXTRA_ARGS="--coadapt-ae" \
  OUTDIR=/outputs/llm_distill/runs/qwen06w_expd_p1 \
  sbatch slurm/train_joint.sbatch

# Phase 2: LoRA fine-tune {llm,ae,vit} on real LIBERO, anchor off
STEPS=12000 INIT=/outputs/llm_distill/runs/qwen06w_expd_p1/latest.pt \
  EXTRA_ARGS="--lora-finetune --lora-targets llm,ae,vit" \
  OUTDIR=/outputs/llm_distill/runs/qwen06w_expd \
  sbatch slurm/train_joint.sbatch
# -> the flexible base (~9.2% on its own; do NOT stop here)
```
(Consult `train_joint.py --help` for the exact co-adapt flag name in your checkout; the driver
supports student-only, co-adapt, and LoRA-finetune modes.)

## 5. Stage 3 — on-policy DAgger + LoRA-adapted action expert (→ 47.5%)

```bash
BASE=/outputs/llm_distill/runs/qwen06w_expd/step_12000.pt

# 5a. collect student-visited states closed-loop, then relabel with the teacher
CKPT=$BASE OUT=/outputs/llm_distill/dagger/expd \
  sbatch slurm/dagger_collect_relabel.sbatch
# -> .../dagger/expd/collect , .../dagger/expd/relabel  (~3.3-3.9k relabeled states)

# 5b. merge base -> plain init, then fine-tune student LLM + AE-LoRA on 50/50 on-policy/demo mix
BASE=$BASE ADAPT_AE=lora STEPS=6000 \
  RELABEL=/outputs/llm_distill/dagger/expd/relabel \
  OUT=/outputs/llm_distill/dagger/expd/train_aelora \
  sbatch slurm/dagger_merge_train.sbatch
# -> .../dagger/expd/train_aelora/step_6000.pt   (the 47.5% deliverable)
```

Ablations reproduced in the report: `ADAPT_AE=none` (frozen AE → 16.67%), and a naive round-2
(re-collect from the 47.5% policy, retrain 6k on only the new pool → 31.67%, regresses).

## 6. Closed-loop gate (120 rollouts)

```bash
CKPT=/outputs/llm_distill/dagger/expd/train_aelora/step_6000.pt
for S in libero_spatial libero_object libero_goal libero_10; do
  STUDENT_CKPT=$CKPT SUITE=$S TASK_IDS="0 1 2 3 4 5 6 7 8 9" N_EP=3 \
    OUTDIR=/outputs/llm_distill/dagger/expd/gate2_aelora/$S \
    sbatch slurm/eval_joint.sbatch
done
# aggregate with: WORK=$WORK bash slurm/dagger_status.sh
# teacher 100% baseline: TEACHER_BASELINE=1 sbatch slurm/eval_joint.sbatch
```

## 7. Compute-benefit benchmark (teacher vs qwen06w backbone)

```bash
# inside the SIF, one idle GPU:
apptainer exec --rocm --cleanenv --bind $WORK/llm_distill:/work --pwd /work \
  --env HIP_VISIBLE_DEVICES=0 --env PYTHONPATH=/work \
  $WORK/images/molmoact2-lerobot-eval.sif python /work/bench_backbone.py --seq 479 --batch 1
# reports params / weight-MB / latency / peak-mem for both backbones + ratios (see success report §4)
```

## File map

| Path | Role |
|---|---|
| `student.py` / `warmstart.py` | `qwen06w` student + Qwen3-0.6B warm-start |
| `joint_patch.py` | attach student to policy; AE LoRA/full/none; hybrid `flow + KV-anchor` loss |
| `data.py` | real LIBERO batches / teacher fused-embed capture |
| `train_joint.py` | trainer driver: functional / co-adapt / LoRA-finetune / DAgger modes |
| `collect_dagger.py` | roll out student, snapshot on-policy states |
| `relabel_teacher.py` | relabel states with the teacher's current-step action (+ `--validate`) |
| `dagger_data.py` | 50/50 on-policy/demo mixed iterator (shape-bucketed) |
| `merge_base.py` | fold legacy/coadapt/lora ckpt into a plain student+AE init |
| `eval_joint.py` | closed-loop LIBERO gate (student or `--teacher-baseline`) |
| `bench_backbone.py` | teacher vs student backbone latency/memory micro-benchmark |
| `validate_student.py` / `validate_joint.py` | module validation |
| `slurm/*.sbatch`, `slurm/dagger_status.sh` | portable Slurm + Apptainer launchers |
| `docker/` | eval image Dockerfile + SIF build recipe + Apptainer contract |
| `curriculum.py`, `minitron_*.py`, `train_reduced.py`, `eval_reduced.py`, `report_expD.py` | deprioritized exploratory tracks (kept for the record) |
