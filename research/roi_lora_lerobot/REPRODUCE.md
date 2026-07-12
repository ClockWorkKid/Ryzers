# Reproducing the ROI-pruning LoRA experiments (MolmoAct2 / LeRobot)

Standalone commands for the four experiments in the reports. Each trains a single-GPU,
resume-aware segment in the training container and evaluates closed-loop LIBERO in the
eval container. The pruning code is the bind-mounted overlay in `docker/roi_overlay/`
(no image rebuild needed after overlay edits).

| # | Experiment | Script | Report |
|---|------------|--------|--------|
| 1 | seam-6 gate_distill, Stage-1 only (scatter-back) | `scripts/roi_fastv_train.sh` (`FASTV=false`) | FEEDBACK §9 |
| 2 | seam-6 gate_distill + FastV, two-stage (L0=3) | `scripts/roi_fastv_train.sh` | FEEDBACK §9.2 |
| 3 | deeper FastV probe — matched L0=9 retrain | `scripts/roi_fastv_train.sh` (`FASTV_L0=9`) | FEEDBACK §9.3 / CAUSAL |
| 4 | causal predictive gating + FastV | `scripts/roi_predict_train.sh` | CAUSAL |

## 0. Prerequisites

```bash
# 0a. Build the images (once per cluster; ROI overlay is baked in + also bind-mounted)
cd docker
docker build -t molmoact2-lerobot-train:rocm942 -f Dockerfile.instinct-gfx942 .
docker build -t molmoact2-lerobot-eval:rocm942  -f Dockerfile.instinct-gfx942-eval .

# 0b. Point the scripts at a scratch/shared filesystem for HF cache + outputs.
#     Every script honours these env vars (defaults shown):
export WORK=$HOME/molmoact2          # -> $WORK/hf_cache and $WORK/outputs
# OV defaults to this repo's docker/roi_overlay (nested layout), auto-detected
# relative to each script; override only to test a work-in-progress overlay.
```

All scripts are single-GPU (`GPU=<id>`), wall-clock capped (`SEG_SECONDS`, default
6600 s < a 2 h partition), and resume from `checkpoints/last` — re-run the same command
(or chain segments with your scheduler's `--dependency=afterany`) until step `STEPS`.

Run a fast correctness smoke first (imports + build + a few real steps + finite loss;
the FastV/predict cut lines print under `ROI_PRUNE_DEBUG`):

```bash
bash scripts/roi_fastv_smoke.sh        # two-stage path (4 steps)
bash scripts/roi_predict_smoke.sh      # causal predictive path (8 steps, checks past-frame load + agreement)
```

## 1. seam-6 gate_distill, Stage-1 only (scatter-back)

Learned mid-ViT gate (seam=6) drops 50 % of ViT patches and scatter-backs a mask token,
so the LLM still sees the full pooled grid (ViT compute saved only). 6000 steps.
keep=0.50 ≈ lossless (97.5 % on the 4 standard suites); sweep `ROI_KEEP` for 0.25/0.75.

```bash
FASTV=false ROI_KEEP=0.50 SEAM=6 STEPS=6000 GPU=0 bash scripts/roi_fastv_train.sh
# -> outputs/roi_gate_keep050
```

## 2. seam-6 gate_distill + FastV (two-stage, L0=3)

Stage-1 gate (keep 50 % patches, scatter-back) + Stage-2 FastV cut after decoder layer
L0=3 that physically drops image tokens down to `FASTV_KEEP`. Last-4 decoder layers
unfrozen; keep 0.9→target curriculum over 4000 steps; 12000 steps.

```bash
# keep 0.50 (≈lossless, 96.5 % on 4 suites)  -> outputs/roi_fastv_keep050_fv050
ROI_KEEP=0.50 FASTV_KEEP=0.50 FASTV_L0=3 STEPS=12000 GPU=0 bash scripts/roi_fastv_train.sh
# keep 0.25 (84.0 % on 4 suites at L0=3)      -> outputs/roi_fastv_keep050_fv025
ROI_KEEP=0.50 FASTV_KEEP=0.25 FASTV_L0=3 STEPS=12000 GPU=1 bash scripts/roi_fastv_train.sh
```

## 3. Deeper FastV probe — matched L0=9 retrain

Same two-stage recipe but the FastV cut is delayed to L0=9, recovering the aggressive
25 %-keep regime (82.0→90.5 % eval-only; the matched retrain locks it in). The `_l0_9`
suffix keeps this run dir distinct from the L0=3 one.

```bash
ROI_KEEP=0.50 FASTV_KEEP=0.25 FASTV_L0=9 STEPS=12000 RUN_SUFFIX=_l0_9 GPU=0 \
  bash scripts/roi_fastv_train.sh
# -> outputs/roi_fastv_keep050_fv025_l0_9
```

## 4. Causal predictive gating + FastV

Identical two-stage pipeline, but the gate that selects the ViT keep-set is scored on the
**past** frame (`H=10` steps earlier) and supervised by the current-frame teacher — so at
inference it is strictly causal and teacher-free (past keep-set prunes the current frame).
Needs the temporal-split processor (in the overlay, auto-mounted).

```bash
# Pair A: L0=3, keep 0.50 -> outputs/roi_predict_keep050_fv050_l03  (97.5 %)
SEL=gate_predict H=10 ROI_KEEP=0.50 FASTV_KEEP=0.50 FASTV_L0=3 STEPS=12000 GPU=0 \
  bash scripts/roi_predict_train.sh
# Pair B: L0=9, keep 0.25 -> outputs/roi_predict_keep050_fv025_l09  (95.2 %)
SEL=gate_predict H=10 ROI_KEEP=0.50 FASTV_KEEP=0.25 FASTV_L0=9 STEPS=12000 GPU=1 \
  bash scripts/roi_predict_train.sh
```

## 5. Closed-loop LIBERO evaluation

`scripts/roi_eval.sh` runs `lerobot-eval` over `RUNS` × `SUITES` × tasks (`N_EP` episodes
each, resumable at task level, `WPG` workers/GPU). `RUNS` = space-separated
`label:rundir:ckptstep` triples (`rundir` relative to `$OUT`, `ckptstep` zero-padded).

Two regimes — set the same `roi_eval.sh` twice into **distinct** `RESDIR`s to get the
train/eval delta:

```bash
# gate-only readout (LLM sees full pooled grid): ROI_FASTV_INFER unset
RUNS="fv050:roi_fastv_keep050_fv050:012000" \
  RESDIR=/outputs/eval_fv050_gateonly NGPU=8 WPG=2 N_EP=10 bash scripts/roi_eval.sh

# FastV-wired (gate drives the in-LLM cut at L0, matches training):
RUNS="fv050:roi_fastv_keep050_fv050:012000" ROI_FASTV_INFER=1 \
  RESDIR=/outputs/eval_fv050_wired NGPU=8 WPG=2 N_EP=10 bash scripts/roi_eval.sh

# causal predictive checkpoint (same flags; processor overlay auto-mounted):
RUNS="pred025:roi_predict_keep050_fv025_l09:012000" ROI_FASTV_INFER=1 \
  RESDIR=/outputs/eval_pred025 NGPU=8 WPG=2 N_EP=10 bash scripts/roi_eval.sh
```

Results land in `$OUT/<RESDIR>/results.csv` (per `label,suite,task,pc_success,...`);
a `_ROI_EVAL_DONE` marker is touched when every unit has an `eval_info.json`.

> Rendering: `MUJOCO_GL=osmesa` (CPU/llvmpipe) is the default and is **required** on
> compute-only datacenter GPUs (EGL fails on gfx942); override to `egl` on a
> graphics-capable GPU. See `SIM_EVAL_PIPELINE.md` for the full rendering rationale.

## 6. Config knobs (all env-overridable)

| env | default | meaning |
|-----|---------|---------|
| `SEL` | `gate_distill` | gate supervision: `gate_distill` (current-frame teacher) or `gate_predict` (past-frame, causal) |
| `ROI_KEEP` | `0.50` | Stage-1 ViT patch keep fraction (scatter-back) |
| `SEAM` | `6` | ViT block after which the gate scores/prunes |
| `FASTV` | `true` | (`roi_fastv_train.sh`) enable Stage-2 FastV; `false` = Stage-1 gate only |
| `FASTV_KEEP` | `0.25` | Stage-2 image-token keep target after the in-LLM cut |
| `FASTV_L0` | `3` | decoder layer after which FastV cuts |
| `UNFREEZE` | `4` | last-N decoder layers full-FT (on top of LoRA-VLM) |
| `CURR_STEPS`/`CURR_START` | `4000`/`0.9` | anneal keep from start→target over N forwards (0=off) |
| `H` | `10` | (`roi_predict_train.sh`) predictive horizon = replan stride |
| `STEPS` | `12000` | total training steps (6000 for exp 1) |
| `GATE_LR`/`DW` | `1e-2`/`20` | gate LR and distillation weight |
| `RUN_SUFFIX` | — | appended to the run dir name (e.g. `_l0_9`) |
